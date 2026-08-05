#!/usr/bin/env python3
"""
vLLM 模型部署脚本 - 支持单卡/多卡/张量并行/流水线并行
适配 Qwen/DeepSeek 系列及多模态模型，支持 YAML 配置驱动

用法:
    # 单卡部署 7B 模型
    python deploy_server.py --model Qwen/Qwen2.5-7B-Instruct --gpu-util 0.9

    # 使用配置文件
    python deploy_server.py --model Qwen/Qwen2.5-7B-Instruct --config configs/vllm_serve.yaml

    # 多卡张量并行部署 32B 模型
    python deploy_server.py --model Qwen/Qwen2.5-32B-Instruct --tensor-parallel 4

    # 部署 AWQ 量化模型
    python deploy_server.py --model ./models/Qwen2.5-7B-AWQ --quantization awq

    # 部署 FP8 模型 (H100+)
    python deploy_server.py --model ./models/DeepSeek-R1-14B-FP8 --quantization fp8 --kv-dtype fp8

    # 多模态模型部署
    python deploy_server.py --model Qwen/Qwen2.5-VL-7B-Instruct --multimodal

    # DeepSeek-V3 多节点部署
    python deploy_server.py --model deepseek-ai/DeepSeek-V3 --tensor-parallel 8 --pipeline-parallel 2 --enable-expert-parallel
"""

import argparse
import json
import os
import subprocess
import sys

import yaml


CONFIG_KEY_MAP = {
    "quantization": "quantization",
    "kv-cache-dtype": "kv_dtype",
    "dtype": "dtype",
    "tensor-parallel-size": "tensor_parallel",
    "pipeline-parallel-size": "pipeline_parallel",
    "enable-expert-parallel": "enable_expert_parallel",
    "gpu-memory-utilization": "gpu_util",
    "max-model-len": "max_model_len",
    "max-num-seqs": "max_num_seqs",
    "max-num-batched-tokens": "max_num_batched_tokens",
    "swap-space": "swap_space",
    "enable-prefix-caching": "enable_prefix_caching",
    "enable-chunked-prefill": "enable_chunked_prefill",
    "multimodal": "multimodal",
    "max-images": "max_images",
    "trust-remote-code": "trust_remote_code",
    "tool-call-parser": "tool_call_parser",
    "host": "host",
    "port": "port",
    "api-key": "api_key",
}


def detect_gpu_capability():
    """检测首块 GPU 的计算能力，返回 (major, minor)；无 GPU 或 torch 不可用时返回 None"""
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return torch.cuda.get_device_capability(0)
    except Exception:
        return None


def detect_model_quantization(model_path: str):
    """从本地模型目录的 config.json 读取量化方法；HuggingFace 模型 ID 或无量化时返回 None"""
    config_path = os.path.join(model_path, "config.json")
    if not os.path.isfile(config_path):
        return None
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    quant_cfg = cfg.get("quantization_config") or {}
    return quant_cfg.get("quant_method")


def apply_hardware_constraints(args):
    """根据 GPU 计算能力校验并调整部署参数

    V100 (Volta, SM 7.0) 限制:
    - 不支持 bfloat16 -> 强制 float16
    - 不支持 FP8 (需要 SM 9.0+) -> 直接报错
    - AWQ GEMM kernel 需要 SM 7.5+ -> 警告并建议 GPTQ
    """
    capability = detect_gpu_capability()
    if capability is None:
        return
    major, minor = capability
    quant = (args.quantization or "").lower()
    kv_dtype = (args.kv_dtype or "").lower()

    if major < 9 and (quant == "fp8" or kv_dtype.startswith("fp8")):
        print(f"[错误] 当前 GPU (SM {major}.{minor}) 不支持 FP8，需要 Hopper (SM 9.0+)。")
        print("       请改用 GPTQ INT4 / BitsAndBytes NF4 / W8A8。")
        sys.exit(1)

    if major < 8:
        if args.dtype in ("", "auto", "bfloat16"):
            if args.dtype == "bfloat16":
                print(f"[警告] SM {major}.{minor} GPU 不支持 bfloat16，已强制改为 float16")
            args.dtype = "float16"
        if quant == "awq":
            print(f"[警告] AWQ GEMM kernel 需要 SM 7.5+，当前 GPU (SM {major}.{minor}) "
                  "只能使用较慢的 GEMV kernel，建议改用 GPTQ 量化模型。")


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_to_defaults(config: dict) -> dict:
    """将 YAML 配置转换为 argparse 默认值"""
    defaults = {}
    for key, attr in CONFIG_KEY_MAP.items():
        if key in config:
            defaults[attr] = config[key]

    # 处理模型级预设
    model_presets = config.get("model_presets", {})
    defaults["_model_presets"] = model_presets
    return defaults


def build_vllm_command(args) -> list:
    """构建 vLLM 服务启动命令"""
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

    cmd.extend(["--model", args.model])

    if args.dtype:
        cmd.extend(["--dtype", args.dtype])

    if args.quantization:
        cmd.extend(["--quantization", args.quantization])
        if args.quantization == "awq" and not args.dtype:
            cmd.extend(["--dtype", "float16"])

    if args.kv_dtype:
        cmd.extend(["--kv-cache-dtype", args.kv_dtype])

    cmd.extend(["--gpu-memory-utilization", str(args.gpu_util)])

    if args.max_model_len is not None:
        cmd.extend(["--max-model-len", str(args.max_model_len)])

    if args.tensor_parallel > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel)])

    if args.pipeline_parallel > 1:
        cmd.extend(["--pipeline-parallel-size", str(args.pipeline_parallel)])

    if args.enable_expert_parallel:
        cmd.append("--enable-expert-parallel")

    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    if args.multimodal:
        cmd.extend(["--limit-mm-per-prompt", f"image={args.max_images}"])

    if args.trust_remote_code:
        cmd.append("--trust-remote-code")

    if args.tool_call_parser:
        cmd.extend(["--tool-call-parser", args.tool_call_parser])

    if args.swap_space is not None:
        cmd.extend(["--swap-space", str(args.swap_space)])

    if args.max_num_batched_tokens is not None:
        cmd.extend(["--max-num-batched-tokens", str(args.max_num_batched_tokens)])

    if args.max_num_seqs is not None:
        cmd.extend(["--max-num-seqs", str(args.max_num_seqs)])

    if args.api_key:
        cmd.extend(["--api-key", args.api_key])

    cmd.extend(["--host", args.host])
    cmd.extend(["--port", str(args.port)])

    return cmd


def get_model_specific_args(model_name: str, presets: dict = None) -> dict:
    """根据模型名称获取推荐的部署配置"""
    configs = {
        "qwen2.5-7b": {
            "gpu_util": 0.9,
            "max_model_len": 32768,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "qwen2.5-14b": {
            "gpu_util": 0.92,
            "max_model_len": 32768,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "qwen2.5-32b": {
            "gpu_util": 0.92,
            "max_model_len": 32768,
            "tensor_parallel": 2,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "qwen2.5-72b": {
            "gpu_util": 0.92,
            "max_model_len": 32768,
            "tensor_parallel": 4,
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "deepseek-r1-distill-qwen-7b": {
            "gpu_util": 0.9,
            "max_model_len": 8192,
            "trust_remote_code": True,
            "tool_call_parser": "deepseek_v3",
            "enable_prefix_caching": True,
        },
        "deepseek-r1-distill-qwen-14b": {
            "gpu_util": 0.92,
            "max_model_len": 8192,
            "trust_remote_code": True,
            "tool_call_parser": "deepseek_v3",
            "enable_prefix_caching": True,
        },
        "deepseek-r1-distill-qwen-32b": {
            "gpu_util": 0.92,
            "max_model_len": 8192,
            "tensor_parallel": 2,
            "trust_remote_code": True,
            "tool_call_parser": "deepseek_v3",
            "enable_prefix_caching": True,
        },
        "deepseek-v3": {
            "gpu_util": 0.9,
            "max_model_len": 131072,
            "tensor_parallel": 8,
            "pipeline_parallel": 2,
            "enable_expert_parallel": True,
            "quantization": "fp8",
            "kv_dtype": "fp8_e5m2",
            "trust_remote_code": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
        },
        "qwen2.5-vl": {
            "gpu_util": 0.9,
            "max_model_len": 32768,
            "trust_remote_code": True,
            "multimodal": True,
        },
    }

    if presets:
        configs.update(presets)

    model_lower = model_name.lower()
    for key, config in configs.items():
        if key in model_lower:
            return config

    return {}


def apply_config(args, config: dict):
    """将配置文件中的默认值应用到 args（CLI 显式传入的值优先）"""
    defaults = config_to_defaults(config)
    # 模型预设单独保存
    presets = defaults.pop("_model_presets", {})
    args._model_presets = presets

    for attr, value in defaults.items():
        current = getattr(args, attr, None)
        if current is None or current == "":
            setattr(args, attr, value)


def apply_model_specific_args(args):
    """根据模型名称应用预设配置（不覆盖 CLI 显式传入的非默认值）"""
    auto_config = get_model_specific_args(args.model, getattr(args, "_model_presets", None))
    for key, value in auto_config.items():
        attr = key.replace("-", "_")
        current = getattr(args, attr, None)
        # 简单启发式：若当前值为空/None/False/1（默认最小值），则应用预设
        if current in [None, False, 1, ""]:
            setattr(args, attr, value)


def print_deployment_info(args, cmd):
    """打印部署信息"""
    print("=" * 70)
    print("vLLM 模型部署配置")
    print("=" * 70)
    print(f"模型路径:     {args.model}")
    print(f"量化类型:     {args.quantization or '无 (FP16/BF16)'}")
    print(f"KV Cache类型: {args.kv_dtype or '自动 (FP16/BF16)'}")
    print(f"数据类型:     {args.dtype or '自动'}")
    print(f"张量并行:     {args.tensor_parallel}")
    print(f"流水线并行:   {args.pipeline_parallel}")
    print(f"GPU利用率:    {args.gpu_util}")
    print(f"最大序列长度: {args.max_model_len or '模型默认'}")
    print(f"Swap空间:     {args.swap_space or '默认'}")
    print(f"服务地址:     http://{args.host}:{args.port}")
    print(f"OpenAI API:   http://{args.host}:{args.port}/v1/chat/completions")
    print("=" * 70)
    print("启动命令:")
    print(" ".join(cmd))
    print("=" * 70)
    print("\n测试命令示例:")
    print(f'  curl http://{args.host}:{args.port}/v1/models')
    print(f'  curl http://{args.host}:{args.port}/v1/chat/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print(f'    -d \'{{"model": "{args.model}", "messages": [{{"role": "user", "content": "Hello"}}]}}\'')
    print("=" * 70)


def build_parser() -> argparse.ArgumentParser:
    """构建 argparse 解析器"""
    parser = argparse.ArgumentParser(description="vLLM 模型部署工具")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--config", type=str, default="",
                        help="vLLM 服务配置文件路径（如 configs/vllm_serve.yaml）")
    parser.add_argument("--quantization", type=str, default="",
                        choices=["", "awq", "fp8", "gptq", "marlin", "bitsandbytes", "compressed-tensors"],
                        help="量化类型")
    parser.add_argument("--kv-dtype", type=str, default="",
                        choices=["", "fp8", "fp8_e4m3", "fp8_e5m2"],
                        help="KV Cache 数据类型")
    parser.add_argument("--dtype", type=str, default="",
                        choices=["", "auto", "float16", "bfloat16", "float32"],
                        help="模型权重数据类型")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="张量并行大小 (GPU 数量)")
    parser.add_argument("--pipeline-parallel", type=int, default=1,
                        help="流水线并行大小 (节点数量)")
    parser.add_argument("--enable-expert-parallel", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用 MoE 专家并行 (DeepSeek-V3/V4)")
    parser.add_argument("--gpu-util", type=float, default=0.9,
                        help="GPU 内存利用率 (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="最大模型序列长度")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="最大并发序列数")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="每次迭代最大 batched tokens 数")
    parser.add_argument("--swap-space", type=int, default=None,
                        help="CPU 交换空间大小 (GB)")
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用前缀缓存")
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用 Chunked Prefill")
    parser.add_argument("--multimodal", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="启用多模态支持")
    parser.add_argument("--max-images", type=int, default=5,
                        help="每轮对话最大图片数")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="信任远程代码 (Qwen/DeepSeek 部分模型需要)")
    parser.add_argument("--tool-call-parser", type=str, default="",
                        help="工具调用解析器 (如 deepseek_v3)")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8000,
                        help="服务端口")
    parser.add_argument("--api-key", type=str, default="",
                        help="API 密钥")
    parser.add_argument("--auto-config", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="自动根据模型名称选择配置")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印命令，不实际启动")
    return parser


def main():
    parser = build_parser()

    # 第一轮解析，仅用于读取 --config
    first_args, _ = parser.parse_known_args()

    # 如果提供了配置文件，用配置作为默认值
    if first_args.config:
        config = load_config(first_args.config)
        defaults = config_to_defaults(config)
        defaults.pop("_model_presets", None)  # 模型预设稍后单独处理
        parser.set_defaults(**defaults)

    args = parser.parse_args()

    # 再次加载完整配置以获取模型预设
    if args.config:
        full_config = load_config(args.config)
        args._model_presets = full_config.get("model_presets", {})
    else:
        args._model_presets = {}

    # 应用模型自动配置
    if args.auto_config:
        apply_model_specific_args(args)

    # 本地模型未显式指定量化方式时，从 config.json 自动识别
    if not args.quantization:
        detected = detect_model_quantization(args.model)
        if detected:
            args.quantization = detected
            print(f"[自动检测] 从模型配置识别量化方式: {detected}")

    # 根据 GPU 计算能力校验/调整参数 (V100 强制 float16, 拒绝 FP8)
    apply_hardware_constraints(args)

    # 构建命令
    cmd = build_vllm_command(args)

    # 打印信息
    print_deployment_info(args, cmd)

    if args.dry_run:
        print("\n[干运行模式] 不实际启动服务")
        print("等价命令:")
        print("vllm serve " + " ".join(cmd[4:]))
        return

    print("\n[启动服务] ...")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n[服务已停止]")
    except subprocess.CalledProcessError as e:
        print(f"\n[错误] 服务启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
