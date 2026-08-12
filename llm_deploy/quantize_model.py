#!/usr/bin/env python3
"""
模型量化转换脚本 - 支持 AWQ、FP8、GPTQ、W8A8 等多种量化方案
适配 Qwen/DeepSeek 系列大模型，支持 YAML 配置驱动

用法:
    python quantize_model.py --model Qwen/Qwen2.5-7B-Instruct --method awq --output ./models/Qwen2.5-7B-AWQ
    python quantize_model.py --model Qwen/Qwen2.5-7B-Instruct --config configs/awq_4bit.yaml --output ./models/Qwen2.5-7B-AWQ
    python quantize_model.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --method fp8 --output ./models/DeepSeek-R1-14B-FP8
"""

import argparse
import json
import os
import sys
from pathlib import Path

import yaml


DEFAULT_CALIBRATION_TEXTS = [
    "The field of large language models is rapidly evolving.",
    "Quantization helps reduce the computational cost of inference.",
    "AWQ is a post-training quantization method that protects salient weights.",
    "FP8 quantization leverages native Hopper hardware support.",
    "DeepSeek-R1 is a powerful reasoning model.",
    "Qwen models support multiple languages including Chinese and English.",
    "The capital of France is Paris.",
    "Machine learning is a subset of artificial intelligence.",
    "Python is a popular programming language for data science.",
    "SmoothQuant handles activation outliers during quantization.",
]


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(args) -> dict:
    """合并命令行参数与 YAML 配置"""
    config = {}
    if args.config:
        config = load_config(args.config)

    method = args.method or config.get("quantization", {}).get("method", "awq")
    # 方法别名统一
    method_aliases = {
        "smoothquant": "w8a8",
        "int8": "w8a8",
    }
    method = method_aliases.get(method.lower(), method)

    quant_cfg = config.get("quantization", {})

    merged = {
        "method": method,
        "w_bit": args.w_bit if args.w_bit is not None else quant_cfg.get("w_bit", quant_cfg.get("bits", 4)),
        "group_size": args.group_size if args.group_size is not None else quant_cfg.get("q_group_size", quant_cfg.get("group_size", 128)),
        "zero_point": quant_cfg.get("zero_point", True),
        "version": quant_cfg.get("version", "GEMM"),
        "scheme": quant_cfg.get("scheme", "W4A16"),
        "targets": quant_cfg.get("targets", "Linear"),
        "ignore": quant_cfg.get("ignore", ["lm_head"]),
        "weight_format": quant_cfg.get("weight_format", "e4m3"),
        "activation_format": quant_cfg.get("activation_format", "e4m3"),
        "calibration": config.get("calibration", {}),
        "output": config.get("output", {}),
    }
    # GPTQ 专用参数 (llmcompressor / gptqmodel 后端共用同一份 merged config)
    # 不放进上面 dict 是为了保持可读性, 也方便后续扩展更多 GPTQ 选项
    gptq_keys = [
        "gptq_backend",          # llmcompressor (默认) | gptqmodel
        "desc_act", "sym", "static_groups", "true_sequential",
        "block_size", "dampening_frac", "offload_hessians",
        "skip_compression_stats", "batch_size",
        "enable_qwen3_pipeline_patch",
    ]
    for k in gptq_keys:
        if k in quant_cfg:
            merged[k] = quant_cfg[k]
    return merged


def format_calibration_data(tokenizer, texts: list) -> list:
    """使用对话模板格式化校准数据，失败时返回原文本"""
    formatted = []
    for text in texts:
        messages = [{"role": "user", "content": text}]
        try:
            formatted_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted.append(formatted_text)
        except Exception:
            formatted.append(text)
    return formatted


def to_calibration_dataset(formatted_texts: list):
    """把已格式化的文本列表包装成 HuggingFace Dataset, 供 llmcompressor.oneshot 使用

    llmcompressor 的 oneshot 内部会调用 dataset.column_names / dataset.map,
    因此不能直接传 list[str] 或 list[dict], 必须是真正的 datasets.Dataset。
    """
    from datasets import Dataset
    return Dataset.from_list([{"text": t} for t in formatted_texts])


def get_calibration_texts(config: dict) -> list:
    """从配置或默认数据中获取校准文本

    优先级: custom_data(本地JSONL) > dataset(HF数据集) > DEFAULT_CALIBRATION_TEXTS
    """
    calib_cfg = config.get("calibration", {})
    custom_data = calib_cfg.get("custom_data", "")       # 本地 JSONL 路径
    dataset_name = calib_cfg.get("dataset", "")
    num_samples = calib_cfg.get("num_samples", 128)

    # HF 镜像与缓存由 setup_hf_env 在 main() 早期统一设置, 这里不再重复

    # ---- 路径 0: 本地自定义 JSONL (最高优先级) ----
    if custom_data:
        jsonl_path = custom_data
        # 支持相对于项目根目录的路径
        if not os.path.isfile(jsonl_path):
            # 尝试相对于项目根目录 (llm_deploy/ 的父目录)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            alt_path = os.path.join(project_root, jsonl_path)
            if os.path.isfile(alt_path):
                jsonl_path = alt_path
        if os.path.isfile(jsonl_path):
            try:
                import json
                texts = []
                with open(jsonl_path, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= num_samples:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        messages = obj.get("messages")
                        if messages:
                            texts.append(messages)       # → list[dict], 走 apply_chat_template
                        else:
                            text = obj.get("text", "")
                            if text:
                                texts.append(text)       # → str, 走纯文本格式化
                if texts:
                    print(f"[校准] 从 {jsonl_path} 加载了 {len(texts)} 条自定义校准数据")
                    return texts
                print(f"[警告] 自定义校准文件 {jsonl_path} 无有效样本, 回退")
            except Exception as e:
                print(f"[警告] 读取自定义校准文件失败: {e}, 回退")
        else:
            print(f"[警告] custom_data 文件不存在: {custom_data}, 回退")

    # ---- 路径 1: HF 数据集 ----
    if dataset_name:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset_name, split="train")
            texts = []
            for i, sample in enumerate(ds):
                if i >= num_samples:
                    break
                messages = sample.get("messages")
                if messages:
                    texts.append(messages)
                else:
                    text = sample.get("text", "")
                    if text:
                        texts.append(text)
            if texts:
                return texts
            print(f"[警告] 校准数据集 {dataset_name} 未返回有效样本，使用默认数据")
        except Exception as e:
            print(f"[警告] 无法加载校准数据集 {dataset_name}: {e}，使用默认数据")

    # num_samples <= 0 时返回空列表无意义, 至少保证 1 条用于下游 calib_texts[0]
    fallback_n = max(num_samples, 1) if num_samples else 1
    return DEFAULT_CALIBRATION_TEXTS[:fallback_n]


def save_quant_config(output_path: str, config: dict):
    """保存量化配置元数据，方便部署脚本读取"""
    os.makedirs(output_path, exist_ok=True)
    config_path = os.path.join(output_path, "quantize_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"[配置] 量化配置已保存: {config_path}")


def setup_hf_env(config: dict):
    """在调用任何 HF/datasets/llmcompressor 之前设置环境变量

    必须在 import llmcompressor 之前调用, 否则 oneshot 内部 load_dataset
    会去访问 huggingface.co 而非镜像, 在无外网环境超时。
    """
    calib_cfg = config.get("calibration", {})
    hf_endpoint = calib_cfg.get("hf_endpoint", "")
    hf_cache = calib_cfg.get("hf_cache", "")
    offline = calib_cfg.get("hf_offline", False)

    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
        print(f"[env] HF_ENDPOINT={hf_endpoint}")
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache, "datasets")
        print(f"[env] HF_HOME={hf_cache}")
    if offline:
        # 数据已在缓存时强制离线, 避免任何网络 HEAD 请求
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        print("[env] HF_HUB_OFFLINE=1 (离线模式)")


def detect_gpu_capability():
    """检测首块 GPU 的计算能力，返回 (major, minor)；无 GPU 或 torch 不可用时返回 None"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def check_hardware_compatibility(method: str):
    """根据 GPU 计算能力校验量化方法是否可用

    与 deploy_server.apply_hardware_constraints 对齐，在量化前拦截会在部署时
    失败或严重降速的方案，避免用户浪费数小时校准时间。

    V100 (Volta, SM 7.0) 限制:
    - 不支持 FP8 (需要 SM 9.0+) -> 直接报错
    - AWQ GEMM kernel 需要 SM 7.5+ -> 警告并建议 GPTQ (不阻止, 模型仍可加载)
    """
    capability = detect_gpu_capability()
    if capability is None:
        return
    major, _ = capability
    method = method.lower()

    if major < 9 and method == "fp8":
        print(f"[错误] 当前 GPU (SM {capability[0]}.{capability[1]}) 不支持 FP8，需要 Hopper (SM 9.0+)。")
        print("       请改用 GPTQ INT4 / BitsAndBytes NF4 / W8A8。")
        sys.exit(1)

    if major < 8 and method == "awq":
        print(f"[警告] AWQ GEMM kernel 需要 SM 7.5+，当前 GPU (SM {capability[0]}.{capability[1]}) "
              "只能使用较慢的 GEMV kernel。")
        print("       建议改用 GPTQ 量化 (--method gptq) 以获得更好性能。")


def quantize_awq_llmcompressor(model_path: str, output_path: str, config: dict):
    """使用 llm-compressor 进行 AWQ W4A16 量化（推荐方案）"""
    print(f"[AWQ/llmcompressor] 开始量化模型: {model_path}")

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.transformers import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"错误: 未安装 llm-compressor: {e}")
        print("       将回退到 legacy autoawq（已标记 DEPRECATED）。")
        return False

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # 准备校准数据
    calib_texts = get_calibration_texts(config)
    if isinstance(calib_texts[0], list):
        # 已经是 messages 格式
        calib_data = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in calib_texts
        ]
    else:
        calib_data = format_calibration_data(tokenizer, calib_texts)

    # llmcompressor.oneshot 要求 dataset 是 HuggingFace Dataset (会调用 column_names/map)
    calib_dataset = to_calibration_dataset(calib_data)

    recipe = QuantizationModifier(
        targets=config.get("targets", "Linear"),
        scheme=config.get("scheme", "W4A16"),
        ignore=config.get("ignore", ["lm_head"]),
    )

    print("[AWQ/llmcompressor] 执行量化...")
    oneshot(model=model, recipe=recipe, dataset=calib_dataset, num_calibration_samples=len(calib_data))

    os.makedirs(output_path, exist_ok=True)
    # skip_compression_stats=True: 阻止自动推断稀疏度写入 config.json, 与 GPTQ 分支一致
    model.save_pretrained(
        output_path,
        skip_compression_stats=config.get("skip_compression_stats", True),
    )
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)
    print(f"[AWQ/llmcompressor] 量化完成! 模型已保存到: {output_path}")
    return True


def quantize_awq_legacy(model_path: str, output_path: str, config: dict):
    """使用 legacy AutoAWQ 进行 AWQ 量化（兼容性回退）"""
    print(f"[AWQ/legacy] 开始量化模型: {model_path}")

    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except ImportError:
        print("错误: 请安装 autoawq: pip install autoawq")
        sys.exit(1)

    quant_config = {
        "zero_point": config.get("zero_point", True),
        "q_group_size": config.get("group_size", 128),
        "w_bit": config.get("w_bit", 4),
        "version": config.get("version", "GEMM"),
    }

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(
        model_path, device_map="auto", safetensors=True, trust_remote_code=True
    )

    calib_texts = get_calibration_texts(config)
    if isinstance(calib_texts[0], list):
        calib_data = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in calib_texts
        ]
    else:
        calib_data = format_calibration_data(tokenizer, calib_texts)

    print("[AWQ/legacy] 执行量化...")
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)

    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(
        output_path,
        safetensors=config.get("output", {}).get("safetensors", True),
        shard_size=config.get("output", {}).get("shard_size", "4GB"),
    )
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)
    print(f"[AWQ/legacy] 量化完成! 模型已保存到: {output_path}")


def quantize_awq(model_path: str, output_path: str, config: dict):
    """AWQ 量化入口：优先 llm-compressor，失败则回退 legacy"""
    ok = quantize_awq_llmcompressor(model_path, output_path, config)
    if not ok:
        quantize_awq_legacy(model_path, output_path, config)


def quantize_fp8_with_llmcompressor(model_path: str, output_path: str, config: dict):
    """FP8 量化 - 使用 llm-compressor（推荐方案）"""
    print(f"[FP8] 开始量化模型: {model_path}")

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        from llmcompressor.transformers import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("错误: 请安装 llm-compressor: pip install llmcompressor")
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    recipe = QuantizationModifier(
        targets=config.get("targets", "Linear"),
        scheme=config.get("scheme", "FP8_DYNAMIC"),
        ignore=config.get("ignore", ["lm_head"]),
    )

    print("[FP8] 执行 FP8 量化...")
    oneshot(model=model, recipe=recipe)

    print("[FP8] 验证量化后模型...")
    input_ids = tokenizer("Hello my name is", return_tensors="pt").input_ids.to(model.device)
    output = model.generate(input_ids, max_new_tokens=20)
    print(f"[FP8] 示例输出: {tokenizer.decode(output[0])}")

    os.makedirs(output_path, exist_ok=True)
    # skip_compression_stats=True: 阻止 save_pretrained 自动推断稀疏度写入 config.json,
    # 与 GPTQ 分支保持一致
    model.save_pretrained(
        output_path,
        skip_compression_stats=config.get("skip_compression_stats", True),
    )
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)
    print(f"[FP8] 量化完成! 模型已保存到: {output_path}")


def quantize_gptq(model_path: str, output_path: str, config: dict):
    """GPTQ 量化 - 根据 gptq_backend 选 llmcompressor 或 gptqmodel 后端

    后端选择 (实测结论):
      - gptqmodel (V100 推荐): 产出标准 GPTQ 格式 (quant_method=gptq),
        vLLM GPTQConfig.get_min_capability()=60, V100 (SM 7.0) 走 Exllama kernel.
      - llmcompressor (A100+ 推荐): 产出 compressed-tensors 格式,
        vLLM W4A16 scheme.get_min_capability()=80, V100 直接报错, 只能在 A100+ 用.

    部署: gptqmodel 路径 -> --quantization gptq
          llmcompressor 路径 -> --quantization compressed-tensors
    """
    # 配置项 gptq_backend: "gptqmodel" (V100 默认/推荐) | "llmcompressor" (A100+)
    # 不显式配置时默认 llmcompressor (兼容旧配置), V100 用户应在 yaml 里写 gptqmodel
    backend = config.get("gptq_backend", "llmcompressor").lower()
    if backend == "gptqmodel":
        quantize_gptq_with_gptqmodel(model_path, output_path, config)
    else:
        ok = quantize_gptq_with_llmcompressor(model_path, output_path, config)
        if not ok:
            print("[GPTQ] llmcompressor 路径失败, 回退到 gptqmodel")
            quantize_gptq_with_gptqmodel(model_path, output_path, config)


def quantize_gptq_with_llmcompressor(model_path: str, output_path: str, config: dict) -> bool:
    """使用 llmcompressor GPTQModifier 进行 W4A16 量化 (推荐, V100 兼容)"""
    print(f"[GPTQ/llmcompressor] 开始量化模型: {model_path}")

    try:
        from llmcompressor.modifiers.quantization import GPTQModifier
        from llmcompressor.transformers import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"错误: 未安装 llm-compressor: {e}")
        return False

    # device_map="auto" 让 accelerate 把模型分布到多卡; 不加会加载到 CPU,
    # GPTQ 逐层 Hessian 计算会慢到无法实用
    # low_cpu_mem_usage=True: 避免一次性把整个模型载入 CPU 内存再分配到 GPU,
    # 减少加载阶段的内存峰值 (8B 模型 fp16 约 16GB)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto",
        low_cpu_mem_usage=True, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Qwen3 兼容补丁: 让 layer_sequential pipeline 走通, 否则退化到 basic pipeline
    # (basic 下每 batch 重新量化全部 module 且 Hessian 每 batch 清零, 最终只用 1-sample
    # Hessian, 速度 734s/it 且精度劣化). 默认开启, 可在 yaml 里关闭.
    enable_qwen3_patch = config.get("enable_qwen3_pipeline_patch", True)
    patch_installed = False
    if enable_qwen3_patch:
        try:
            from qwen3_pipeline_patch import install_qwen3_pipeline_patch
            patch_installed = install_qwen3_pipeline_patch(model)
            if patch_installed:
                print("[GPTQ/llmcompressor] 已启用 Qwen3 layer_sequential 兼容补丁 "
                      "(position_embeddings 缓存 hook)")
        except Exception as e:
            print(f"[GPTQ/llmcompressor] Qwen3 patch 安装失败 (忽略, 回退默认 pipeline): {e}")

    # 准备校准数据 -> 包装成 HuggingFace Dataset
    calib_texts = get_calibration_texts(config)
    if isinstance(calib_texts[0], list):
        calib_data = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in calib_texts
        ]
    else:
        calib_data = format_calibration_data(tokenizer, calib_texts)
    calib_dataset = to_calibration_dataset(calib_data)

    # basic pipeline (Qwen3 等新架构会自动降级) 下 Hessian 累积在 GPU 上
    # 会放大数值误差, offload_hessians=True 把 Hessian 移到 CPU, 用显存换精度
    offload_hessians = config.get("offload_hessians", True)

    recipe = GPTQModifier(
        targets=config.get("targets", "Linear"),
        scheme=config.get("scheme", "W4A16"),
        ignore=config.get("ignore", ["lm_head"]),
        block_size=config.get("block_size", 128),
        dampening_frac=config.get("dampening_frac", 0.01),
        offload_hessians=offload_hessians,
    )

    print(f"[GPTQ/llmcompressor] scheme={config.get('scheme', 'W4A16')}, "
          f"校准样本数={len(calib_data)}, offload_hessians={offload_hessians}, "
          f"skip_compression_stats={config.get('skip_compression_stats', True)}")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_dataset,
        num_calibration_samples=len(calib_data),
    )

    os.makedirs(output_path, exist_ok=True)
    # skip_compression_stats=True: 阻止 save_pretrained 扫描模型稀疏度并写入
    # config.json 的 sparsity_config 字段. 否则 llmcompressor 会推断稀疏度,
    # vLLM 加载时可能按稀疏模型处理, 偏离 W4A16 量化语义.
    model.save_pretrained(
        output_path,
        skip_compression_stats=config.get("skip_compression_stats", True),
    )
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)

    # 卸载 Qwen3 兼容补丁, 避免影响后续推理 (hook 引用了闭包, 显式 remove 更干净)
    if patch_installed:
        try:
            from qwen3_pipeline_patch import uninstall_qwen3_pipeline_patch
            uninstall_qwen3_pipeline_patch(model)
        except Exception:
            pass

    print(f"[GPTQ/llmcompressor] 量化完成! 模型已保存到: {output_path}")
    print(f"[GPTQ/llmcompressor] 部署: deploy_server.py --model {output_path} "
          f"--quantization compressed-tensors --dtype float16")
    return True


def quantize_gptq_with_gptqmodel(model_path: str, output_path: str, config: dict):
    """使用 GPTQModel 进行 GPTQ 量化 (回退路径, 产出 gptq 格式)

    gptqmodel 2.0 的 MODEL_MAP 不含 qwen3, 量化 Qwen3 模型前必须先安装
    qwen3_gptq_adapter (注入 Qwen3GPTQ, 复用 Qwen2GPTQ 结构). adapter 是幂等的,
    非 Qwen3 模型调用 install 也不会有副作用 (只是没人查 model_type).
    """
    print(f"[GPTQ/gptqmodel] 开始量化模型: {model_path}")

    try:
        from gptqmodel import GPTQModel, QuantizeConfig
        from transformers import AutoTokenizer
    except ImportError:
        print("错误: 请安装 gptqmodel: pip install gptqmodel")
        sys.exit(1)

    # Qwen3 兼容: 在 GPTQModel.from_pretrained 之前注入 Qwen3GPTQ
    # (gptqmodel 2.0 MODEL_MAP 没有 qwen3, 不注入会 TypeError: qwen3 isn't supported yet)
    try:
        from qwen3_gptq_adapter import install_qwen3_gptq_adapter
        if install_qwen3_gptq_adapter():
            print("[GPTQ/gptqmodel] 已安装 qwen3_gptq_adapter (Qwen3 -> Qwen2GPTQ 结构映射)")
    except Exception as e:
        print(f"[GPTQ/gptqmodel] qwen3_gptq_adapter 安装失败 (非 Qwen3 模型可忽略): {e}")

    quant_config = QuantizeConfig(
        bits=config.get("w_bit", 4),
        group_size=config.get("group_size", 128),
        desc_act=config.get("desc_act", True),
        sym=config.get("sym", True),
        static_groups=config.get("static_groups", False),
    )

    model = GPTQModel.from_pretrained(model_path, quant_config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    calib_texts = get_calibration_texts(config)
    if isinstance(calib_texts[0], list):
        formatted = [
            tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in calib_texts
        ]
    else:
        formatted = format_calibration_data(tokenizer, calib_texts)

    # gptqmodel 2.0 quantize() 接受 List[str] | List[int] | List[Dict[str, Tensor]].
    # 实测直接传 List[Tensor(1, seq_len)] 会触发 IndexError
    #   (prepare_dataset 做 example["input_ids"], 对裸 2D tensor 是非法索引).
    # 最稳妥: 传 List[str] + tokenizer 参数, 让 gptqmodel 内部 tokenize.
    # tokenizer 会经 Tokenicer 包装 (trust_remote_code 透传), 保持与 from_pretrained 一致.
    print(f"[GPTQ/gptqmodel] 执行量化 (校准样本: {len(formatted)} 条 str, batch_size={config.get('batch_size', 1)})...")
    model.quantize(
        formatted,
        batch_size=config.get("batch_size", 1),
        tokenizer=tokenizer,
    )

    os.makedirs(output_path, exist_ok=True)
    model.save_quantized(output_path)
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)
    print(f"[GPTQ/gptqmodel] 量化完成! 模型已保存到: {output_path}")
    print(f"[GPTQ/gptqmodel] 部署: deploy_server.py --model {output_path} "
          f"--quantization gptq --dtype float16")


def quantize_int8_smoothquant(model_path: str, output_path: str, config: dict):
    """W8A8 SmoothQuant 量化 - 权重和激活都量化到 INT8"""
    print(f"[W8A8] 开始 SmoothQuant 量化: {model_path}")

    try:
        from llmcompressor.modifiers.quantization import QuantizationModifier
        # llmcompressor 新旧版本导出路径不同, 做兼容处理
        try:
            from llmcompressor.modifiers.quantization import SmoothQuantModifier
        except ImportError:
            from llmcompressor.modifiers.smoothquant import SmoothQuantModifier
        from llmcompressor.transformers import oneshot
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("错误: 请安装 llm-compressor")
        sys.exit(1)

    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    calib_texts = get_calibration_texts(config)
    if isinstance(calib_texts[0], list):
        ds = [
            {
                "text": tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=False
                )
            }
            for msgs in calib_texts
        ]
    else:
        ds = [{"text": t} for t in format_calibration_data(tokenizer, calib_texts)]

    # llmcompressor.oneshot 要求 dataset 是 HuggingFace Dataset
    from datasets import Dataset
    calib_dataset = Dataset.from_list(ds)

    recipe = [
        SmoothQuantModifier(smoothing_strength=0.8),
        QuantizationModifier(
            targets=config.get("targets", "Linear"),
            scheme=config.get("scheme", "W8A8"),
            ignore=config.get("ignore", ["lm_head"]),
        ),
    ]

    print("[W8A8] 执行 SmoothQuant 量化...")
    oneshot(
        model=model,
        recipe=recipe,
        dataset=calib_dataset,
        num_calibration_samples=min(len(ds), 128),
    )

    os.makedirs(output_path, exist_ok=True)
    # skip_compression_stats=True: 阻止自动推断稀疏度写入 config.json, 与 GPTQ 分支一致
    model.save_pretrained(
        output_path,
        skip_compression_stats=config.get("skip_compression_stats", True),
    )
    tokenizer.save_pretrained(output_path)
    save_quant_config(output_path, config)
    print(f"[W8A8] 量化完成! 模型已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="大模型量化转换工具")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--method", type=str, default="",
                        choices=["", "awq", "fp8", "gptq", "w8a8", "smoothquant", "int8", "bitsandbytes"],
                        help="量化方法: awq(推荐), fp8(H100+), gptq(通用), w8a8/smoothquant/int8(通用), bitsandbytes(仅部署)")
    parser.add_argument("--config", type=str, default="",
                        help="YAML 配置文件路径（如 configs/awq_4bit.yaml）")
    parser.add_argument("--output", type=str, required=True,
                        help="量化模型输出路径")
    parser.add_argument("--w-bit", type=int, default=None,
                        help="权重量化位宽 (AWQ/GPTQ 默认 4)")
    parser.add_argument("--group-size", type=int, default=None,
                        help="量化分组大小 (默认 128)")
    parser.add_argument("--validate", action="store_true",
                        help="量化后自动执行 PPL 验证 (对比 baseline 与量化模型的 Perplexity)")
    parser.add_argument("--max-ppl-delta", type=float, default=5.0,
                        help="PPL 验证阈值 (默认 5.0, 仅 --validate 时生效)")

    args = parser.parse_args()

    if not args.method and not args.config:
        parser.error("请提供 --method 或 --config 之一")

    config = merge_config(args)

    # 在任何 llmcompressor/datasets import 之前设置 HF 环境 (镜像/缓存/离线)
    setup_hf_env(config)

    print("=" * 60)
    print("大模型量化转换工具")
    print("=" * 60)
    print(f"原始模型: {args.model}")
    print(f"量化方法: {config['method'].upper()}")
    print(f"输出路径: {args.output}")
    print(f"配置: w_bit={config['w_bit']}, group_size={config['group_size']}")
    print("=" * 60)

    method = config["method"]
    if method == "awq":
        check_hardware_compatibility(method)
        quantize_awq(args.model, args.output, config)
    elif method == "fp8":
        check_hardware_compatibility(method)
        quantize_fp8_with_llmcompressor(args.model, args.output, config)
    elif method == "gptq":
        quantize_gptq(args.model, args.output, config)
    elif method == "w8a8":
        quantize_int8_smoothquant(args.model, args.output, config)
    elif method == "bitsandbytes":
        print("\n[BitsAndBytes] 动态量化无需预量化模型。")
        print("              部署时直接使用 deploy_server.py --quantization bitsandbytes")
        print("              或参考 configs/bitsandbytes_nf4.yaml")
        return
    else:
        parser.error(f"不支持的量化方法: {method}")

    # ---- 量化后 PPL 验证 (可选) ----
    if args.validate:
        print("\n" + "=" * 60)
        print("量化后 PPL 验证 (--validate)")
        print("=" * 60)
        # 检查输出目录是否存在 (量化成功的标志)
        if not os.path.isdir(args.output):
            print("[跳过] 量化未完成或输出目录不存在，跳过验证")
        else:
            try:
                # 确定量化方式参数
                quant_method = config.get("gptq_backend", method)
                if quant_method == "gptqmodel":
                    quant_arg = "gptq"
                elif quant_method == "llmcompressor":
                    quant_arg = "compressed-tensors"
                else:
                    quant_arg = method if method != "bitsandbytes" else ""

                from validate_calibration import validate_quantization
                v_result = validate_quantization(
                    baseline_path=args.model,
                    quantized_path=args.output,
                    quantization=quant_arg,
                    dtype=config.get("calibration", {}).get("dtype", "auto"),
                    max_ppl_delta=args.max_ppl_delta,
                    num_samples=min(200, args.group_size or 128),  # 与校准样本数大致匹配
                    verbose=True,
                )
                if v_result["error"]:
                    print(f"\n  [验证跳过] {v_result['error']}")
                elif v_result["passed"]:
                    print(f"\n  ✅ PPL 验证通过 (delta={v_result['delta']:.4f})")
                else:
                    print(f"\n  ⚠️  PPL 偏移 {v_result['delta']:.4f} 超过阈值 {v_result['threshold']}")
                    print(f"      建议检查校准数据质量或增大 num_samples")
                    print(f"      模型仍已导出，可自行决定是否使用")
            except ImportError:
                print("[跳过] 未找到 validate_calibration 模块，跳过验证")
            except Exception as e:
                print(f"[跳过] PPL 验证异常: {e}")

    print("\n量化完成! 下一步: 使用 deploy_server.py 部署模型")


if __name__ == "__main__":
    main()
