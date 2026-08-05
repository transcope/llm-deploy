#!/usr/bin/env python3
"""
量化后 PPL 验证模块

对比 baseline (FP16/BF16) 与量化模型的 Perplexity，校验校准数据质量与量化过程
是否引入了不可接受的精度损失。结果**仅警告不阻断**，由用户自行判断是否重做量化。

PPL 计算直接使用 transformers (AutoModelForCausalLM)，不依赖 lm-eval 的任务系统。
这样可以避免 lm-eval 自定义任务注册问题，同时减少依赖和运行时间。

用法:
    # 由 quantize_model.py --validate 自动调用
    python quantize_model.py --model Qwen/Qwen2.5-7B-Instruct --method gptq \\
        --config configs/gptq_4bit_v100_gptqmodel.yaml --output ./models/Qwen2.5-7B-GPTQ \\
        --validate

    # 独立验证（模型已量化，补跑验证）
    python scripts/validate_calibration.py \\
        --baseline Qwen/Qwen2.5-7B-Instruct \\
        --quantized ./models/Qwen2.5-7B-GPTQ \\
        --quantization gptq

    # 指定验证文本（默认使用内置 200 条通用文本）
    python scripts/validate_calibration.py \\
        --baseline ... --quantized ... \\
        --val-data ./data/validation.jsonl \\
        --num-samples 100
"""

import argparse
import json
import math
import os
import sys
import time


# =====================================================================
# 内置验证文本 — 200 条通用中英文，覆盖日常对话、知识问答、指令遵循等场景
# 当未指定 --val-data 时使用。这些文本确保验证可离线执行，不依赖外部数据集。
# =====================================================================
DEFAULT_VALIDATION_TEXTS = [
    # ---- 中文日常对话 (40) ----
    "你好，今天天气怎么样？",
    "请帮我查一下明天北京的空气质量。",
    "给我讲一个关于人工智能的笑话。",
    "用 Python 写一个快速排序算法。",
    "请解释一下什么是量子计算。",
    "翻译成英文：机器学习是人工智能的一个子集。",
    "列出全球前五大科技公司。",
    "什么是 5G 网络的主要特点？",
    "唐代有哪些著名诗人？",
    "请为我生成一份关于气候变化的演讲稿提纲。",
    "如何提高编程效率？",
    "简述 TCP/IP 协议栈的四层结构。",
    "今天股票市场表现如何？",
    "给我推荐三部值得一看的科幻电影。",
    "什么是区块链技术？",
    "请解释 HTTP 和 HTTPS 的区别。",
    "如何学习一门新语言？",
    "什么是光合作用？",
    "请为我写一首关于秋天的诗。",
    "比较深度学习和机器学习的关系。",
    "什么是 RESTful API？",
    "如何优化数据库查询性能？",
    "请介绍一下 Docker 容器技术。",
    "什么是微服务架构？",
    "简述 Git 的工作流程。",
    "什么是 DevOps？",
    "如何保障网络安全？",
    "请解释什么是边缘计算。",
    "什么是大语言模型？",
    "Transformer 架构的核心创新是什么？",
    "什么是注意力机制？",
    "请解释 BP 算法的原理。",
    "什么是强化学习？",
    "监督学习和无监督学习有什么区别？",
    "什么是过拟合？如何防止？",
    "请解释什么是迁移学习。",
    "什么是生成对抗网络？",
    "请介绍一下自然语言处理的主要任务。",
    "什么是词嵌入？",
    "RNN 和 LSTM 有什么区别？",
    # ---- 中文知识问答 (40) ----
    "中国的首都是哪个城市？",
    "珠穆朗玛峰有多高？",
    "水在标准大气压下的沸点是多少摄氏度？",
    "世界上最大的海洋是哪个？",
    "爱因斯坦提出了哪些重要理论？",
    "什么是 DNA？",
    "请列出太阳系八大行星。",
    "人类最早的文字是什么？",
    "什么是相对论？",
    "青霉素是谁发现的？",
    "第一次工业革命始于哪个国家？",
    "什么是碳中和？",
    "请解释一下 PCR 检测的原理。",
    "什么是 CRISPR 基因编辑技术？",
    "世界上最高的建筑是什么？",
    "什么是暗物质？",
    "请解释一下黑洞的形成。",
    "什么是生态系统？",
    "元素周期表有多少个元素？",
    "什么是光合作用的化学方程式？",
    "人类基因组计划是什么？",
    "什么是疫苗的工作原理？",
    "请解释什么是区块链的共识机制。",
    "什么是云计算？",
    "什么是大数据？",
    "请解释什么是物联网。",
    "什么是 3D 打印技术？",
    "什么是虚拟现实？",
    "什么是增强现实？",
    "请解释一下 5G 和 4G 的主要区别。",
    "什么是 Wi-Fi 6？",
    "什么是 IPv6？",
    "请解释一下什么是 VPN。",
    "什么是 SSL/TLS？",
    "什么是 CDN？",
    "请解释一下什么是 DNS。",
    "什么是 SQL 注入？如何防御？",
    "什么是 XSS 攻击？",
    "请解释一下什么是 OAuth 2.0。",
    "什么是 JWT？",
    # ---- 英文通用 (40) ----
    "The quick brown fox jumps over the lazy dog.",
    "Artificial intelligence is transforming every industry.",
    "Please summarize the main points of the meeting.",
    "Write a function that checks if a string is a palindrome.",
    "What is the capital of France?",
    "Explain the concept of recursion in programming.",
    "How does a transformer model work?",
    "What are the benefits of using cloud computing?",
    "Describe the water cycle in nature.",
    "What is the difference between stack and queue?",
    "Write a SQL query to find duplicate emails in a table.",
    "Explain the principle of least privilege in security.",
    "What is the time complexity of binary search?",
    "Describe how HTTPS ensures secure communication.",
    "What is the difference between TCP and UDP?",
    "Explain what a design pattern is in software engineering.",
    "What is the Singleton pattern?",
    "Describe the Model-View-Controller architecture.",
    "What is dependency injection?",
    "Explain the concept of functional programming.",
    "What is a database index and how does it work?",
    "Describe the ACID properties in databases.",
    "What is the CAP theorem?",
    "Explain the difference between SQL and NoSQL databases.",
    "What is a RESTful web service?",
    "Describe the GraphQL query language.",
    "What is gRPC?",
    "Explain how WebSockets work.",
    "What is serverless computing?",
    "Describe the concept of containerization.",
    "What is Kubernetes?",
    "Explain what a service mesh is.",
    "What is infrastructure as code?",
    "Describe the CI/CD pipeline.",
    "What is test-driven development?",
    "Explain the agile software development methodology.",
    "What is the Scrum framework?",
    "Describe the Kanban method.",
    "What is site reliability engineering?",
    "Explain the concept of chaos engineering.",
    # ---- 英文指令遵循 (40) ----
    "Translate the following sentence to Chinese: 'Machine learning is a subset of artificial intelligence.'",
    "Write a Python script to download all images from a webpage.",
    "Create a regular expression to validate email addresses.",
    "Generate a Dockerfile for a Python Flask application.",
    "Write a bash command to find all files larger than 100MB.",
    "Create a simple REST API endpoint using FastAPI.",
    "Write a SQL query to join three tables.",
    "Implement a binary tree in Python.",
    "Write a function to calculate Fibonacci numbers using dynamic programming.",
    "Create a simple neural network using PyTorch.",
    "Generate a yaml configuration for a Kubernetes deployment.",
    "Write a curl command to test a POST endpoint.",
    "Create a Python decorator that measures execution time.",
    "Write a script to parse JSON logs and extract error messages.",
    "Implement a LRU cache in Python.",
    "Create a simple web scraper using BeautifulSoup.",
    "Write a function to find the longest common subsequence.",
    "Generate a regular expression for matching phone numbers.",
    "Write a Python class for a simple bank account system.",
    "Create a unit test for a sorting function.",
    "Implement a producer-consumer pattern using Python threads.",
    "Write a SQL query to pivot rows to columns.",
    "Create a simple React component for a todo list.",
    "Write a bash one-liner to count lines of code in a project.",
    "Generate a Python script to resize all images in a directory.",
    "Write a function to detect cycles in a directed graph.",
    "Create a simple pub/sub system in Python.",
    "Implement a rate limiter using token bucket algorithm.",
    "Write a Python script to monitor CPU and memory usage.",
    "Create a simple load balancer algorithm.",
    "Write a function to serialize and deserialize a binary tree.",
    "Generate a Python script to split a large CSV file.",
    "Implement a version of merge sort in Python.",
    "Write a function to find the shortest path in a weighted graph.",
    "Create a simple caching proxy server in Python.",
    "Write a Python script to convert Markdown to HTML.",
    "Implement a simple state machine in Python.",
    "Write a function to validate JSON schema.",
    "Create a simple command-line argument parser.",
    "Generate a Python script to backup files to S3.",
    # ---- 领域感知 (40) ----
    "什么是 5G 核心网的 SBA 架构？",
    "请解释 3GPP 定义的网络切片概念。",
    "5G NR 的帧结构是怎样的？",
    "什么是 Massive MIMO？",
    "请解释 OFDM 和 OFDMA 的区别。",
    "什么是边缘计算在 5G 中的应用？",
    "请描述 5G 的三大应用场景。",
    "什么是 VoNR？",
    "请解释 5G SA 和 NSA 架构的区别。",
    "什么是 SDN 和 NFV？",
    "5G 频谱有哪些频段？",
    "什么是波束赋形技术？",
    "请解释 5G 的低延迟特性如何实现。",
    "什么是 URLLC？",
    "请描述 5G 核心网的用户面和控制面分离。",
    "什么是网络数据分析功能？",
    "请解释 5G 的服务化架构。",
    "什么是 AMF、SMF、UPF？",
    "请描述 5G 注册流程。",
    "什么是 PDU 会话？",
    "请解释 5G QoS 模型。",
    "什么是网络暴露功能？",
    "请描述 5G 的安全架构。",
    "什么是 SUPI 和 SUCI？",
    "请解释 5G 的认证流程。",
    "什么是 MEC？",
    "请描述 5G 的定位服务。",
    "什么是 V2X？",
    "请解释 5G 的广播多播服务。",
    "什么是 NR-U？",
    "请描述 5G 的载波聚合技术。",
    "什么是 DU、CU、RU 的拆分架构？",
    "请解释 O-RAN 的概念。",
    "什么是 RIC？",
    "请描述 5G 的能效优化技术。",
    "什么是 NTN？",
    "请解释 5G-Advanced 的新特性。",
    "什么是 RedCap？",
    "请描述 AI/ML 在 5G 网络中的应用。",
    "什么是 6G 的潜在关键技术？",
]


# =====================================================================
# 直接 PPL 计算 (使用 transformers)
# =====================================================================

def compute_ppl(
    model_path: str,
    texts: list[str],
    quantization: str = "",
    dtype: str = "float16",
    max_length: int = 2048,
    stride: int = 512,
    device: str = "cuda",
) -> float:
    """使用 transformers 直接计算模型在文本列表上的平均 PPL

    原理:
        1. 对每条文本用 tokenizer 编码
        2. 滑动窗口计算交叉熵损失
        3. PPL = exp(mean loss)
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # ---- 加载 tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 加载模型 ----
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                   "auto": "auto", "float32": torch.float32}.get(dtype, "auto")

    # 单 GPU 加载 (8B 模型 ~16GB FP16, 单张 V100 32GB 足够)
    # 不使用 device_map="auto" 避免跨 GPU 分发导致 tensor device 冲突
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "device_map": None,
    }

    if quantization:
        if quantization == "gptq":
            model_kwargs["quantization_config"] = {"quantization_method": "gptq"}
        elif quantization == "bitsandbytes":
            model_kwargs["load_in_4bit"] = True
        # other quantizations: let transformers auto-detect from config

    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    model.eval()
    model = model.to(device)

    # ---- 逐条计算 PPL ----
    total_loss = 0.0
    total_tokens = 0
    nlls = []

    for idx, text in enumerate(texts):
        encodings = tokenizer(text, return_tensors="pt", truncation=True,
                              max_length=max_length)
        input_ids = encodings.input_ids.to(device)
        seq_len = input_ids.size(1)

        if seq_len < 10:
            # 太短跳过
            continue

        # 滑动窗口 PPL
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        neg_log_likelihood = 0.0
        n_tokens = 0

        prev_end_loc = 0
        for begin_loc in range(0, seq_len, stride):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc  # 避免重复计算 overlap 部分
            input_chunk = input_ids[:, begin_loc:end_loc]

            with torch.no_grad():
                outputs = model(input_chunk, labels=input_chunk)
                # loss 是 per-token 平均交叉熵
                loss = outputs.loss
                # 转换为 per-token nll
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_chunk[..., 1:].contiguous()
                loss_chunk = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1)
                )
                neg_log_likelihood += loss_chunk.sum().item()
                n_tokens += trg_len - 1  # 减去 label shift 损失的一个位置

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

        avg_nll = neg_log_likelihood / max(n_tokens, 1)
        ppl = math.exp(min(avg_nll, 100))  # cap 防止溢出
        nlls.append(avg_nll)
        total_loss += avg_nll

        if (idx + 1) % 10 == 0:
            print(f"    [{idx+1}/{len(texts)}] text_len={seq_len} "
                  f"current_ppl={ppl:.4f}  avg_ppl_sofar={math.exp(total_loss/(idx+1)):.4f}")

    # 清理显存
    del model
    torch.cuda.empty_cache()

    if not nlls:
        return -1.0

    # 全局平均 PPL: exp(mean of per-text NLL)
    mean_nll = sum(nlls) / len(nlls)
    overall_ppl = math.exp(min(mean_nll, 100))
    return overall_ppl


def load_texts_from_jsonl(path: str, limit: int = 0) -> list[str]:
    """从 JSONL 文件加载文本列表 (每行 {"text": "..."})"""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    texts.append(text)
            except json.JSONDecodeError:
                continue
    if 0 < limit < len(texts):
        texts = texts[:limit]
    return texts


def validate_quantization(
    baseline_path: str,
    quantized_path: str,
    quantization: str = "",
    dtype: str = "auto",
    max_ppl_delta: float = 5.0,
    num_samples: int = 200,
    val_texts: list = None,
    val_data_path: str = "",
    verbose: bool = True,
) -> dict:
    """对比 baseline 与量化模型的 PPL，返回验证结果

    使用 transformers 直接计算 PPL (不依赖 lm-eval 的任务系统)

    参数:
        baseline_path: baseline 模型路径或 HF ID
        quantized_path: 量化模型路径
        quantization: 量化方式 (gptq/awq/compressed-tensors 等)
        dtype: 模型加载精度
        max_ppl_delta: PPL 差异阈值，超过此值则标记为失败
        num_samples: 使用的验证样本数
        val_texts: 自定义验证文本列表 (None 则使用内置文本)
        val_data_path: 自定义验证数据集路径 (优先级高于 val_texts)
        verbose: 是否打印详细信息

    返回:
        {
            "baseline_ppl": float,      # baseline PPL
            "quantized_ppl": float,     # 量化模型 PPL
            "delta": float,              # PPL 差异
            "passed": bool,              # 是否通过验证
            "threshold": float,          # 阈值
            "error": str,                # 错误信息 (成功时为空)
        }
    """
    result = {
        "baseline_ppl": -1.0,
        "quantized_ppl": -1.0,
        "delta": -1.0,
        "passed": False,
        "threshold": max_ppl_delta,
        "error": "",
    }

    # ---- 准备验证文本 ----
    if val_data_path and os.path.isfile(val_data_path):
        texts = load_texts_from_jsonl(val_data_path, limit=num_samples)
        if verbose:
            print(f"  使用验证数据集: {val_data_path}")
    else:
        texts = val_texts if val_texts is not None else DEFAULT_VALIDATION_TEXTS
        texts = texts[:num_samples]
        if verbose:
            print(f"  使用内置验证文本: {len(texts)} 条")

    if not texts:
        result["error"] = "验证文本列表为空"
        return result

    if verbose:
        print(f"  每文本最大长度: 2048 tokens (滑动窗口 stride=512)")

    # ---- 确定 dtype ----
    if dtype == "auto":
        dtype = "float16"  # V100 安全默认值
    elif dtype == "bfloat16":
        # V100 不支持 bfloat16
        import torch
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability()
            if cap[0] < 8:
                if verbose:
                    print(f"  [注意] V100 (SM {cap[0]}.{cap[1]}) 不支持 bfloat16，降级为 float16")
                dtype = "float16"

    try:
        # ---- 运行 baseline PPL ----
        if verbose:
            print(f"\n  [1/2] 计算 baseline PPL ({baseline_path}) ...")
            print(f"        加载精度: {dtype}")
        start = time.time()
        baseline_ppl = compute_ppl(
            model_path=baseline_path,
            texts=texts,
            quantization="",  # baseline 不量化
            dtype=dtype,
        )
        elapsed = time.time() - start
        if baseline_ppl < 0:
            result["error"] = f"baseline PPL 计算失败"
            return result
        result["baseline_ppl"] = baseline_ppl
        if verbose:
            print(f"  ✅ baseline PPL = {baseline_ppl:.4f}  (耗时 {elapsed:.0f}s)")

        # ---- 运行量化模型 PPL ----
        if verbose:
            print(f"\n  [2/2] 计算量化模型 PPL ({quantized_path}) ...")
            print(f"        量化方式: {quantization or '无'}  加载精度: {dtype}")
        start = time.time()
        quantized_ppl = compute_ppl(
            model_path=quantized_path,
            texts=texts,
            quantization=quantization,
            dtype=dtype,
        )
        elapsed = time.time() - start
        if quantized_ppl < 0:
            result["error"] = f"量化模型 PPL 计算失败"
            return result
        result["quantized_ppl"] = quantized_ppl
        if verbose:
            print(f"  ✅ 量化模型 PPL = {quantized_ppl:.4f}  (耗时 {elapsed:.0f}s)")

        # ---- 对比 ----
        delta = quantized_ppl - baseline_ppl
        result["delta"] = delta
        result["passed"] = delta <= max_ppl_delta

        if verbose:
            print(f"\n  {'='*50}")
            print(f"  PPL 验证结果")
            print(f"  {'='*50}")
            print(f"  Baseline (FP16):          ppl = {baseline_ppl:.4f}")
            print(f"  量化模型 ({quantization or 'auto'}):    ppl = {quantized_ppl:.4f}")
            print(f"  Delta:                     {delta:+.4f}  (阈值: {max_ppl_delta})")
            if result["passed"]:
                print(f"  验证: ✅ 通过 (PPL 偏移在合理范围内)")
            else:
                print(f"  验证: ⚠️ 超出阈值 (PPL 偏移过大，建议检查校准数据)")
            print(f"  {'='*50}")

    except Exception as e:
        result["error"] = str(e)
        if verbose:
            print(f"  [错误] 验证过程异常: {e}")
            import traceback
            traceback.print_exc()

    return result


def cli_validate():
    """独立验证 CLI 入口"""
    parser = argparse.ArgumentParser(
        description="量化模型 PPL 验证工具 — 对比 baseline 与量化模型的 Perplexity"
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Baseline 模型路径或 HuggingFace 模型 ID (FP16/BF16 原始模型)")
    parser.add_argument("--quantized", type=str, required=True,
                        help="量化模型路径")
    parser.add_argument("--quantization", type=str, default="",
                        choices=["", "awq", "gptq", "fp8", "compressed-tensors", "marlin", "bitsandbytes"],
                        help="量化模型的量化方式")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="模型加载精度")
    parser.add_argument("--max-ppl-delta", type=float, default=5.0,
                        help="PPL 差异阈值 (默认 5.0)")
    parser.add_argument("--num-samples", type=int, default=200,
                        help="验证样本数 (默认 200)")
    parser.add_argument("--val-data", type=str, default="",
                        help="验证数据集路径 (JSONL, 每行 {\"text\": \"...\"})")
    parser.add_argument("--output", type=str, default="",
                        help="结果输出路径 (JSON)")
    args = parser.parse_args()

    print("=" * 60)
    print("量化后 PPL 验证")
    print("=" * 60)
    print(f"Baseline:  {args.baseline}")
    print(f"量化模型:  {args.quantized}")
    print(f"量化方式:  {args.quantization or '无'}")
    print(f"验证样本:  {args.num_samples}")
    print(f"PPL 阈值:  {args.max_ppl_delta}")
    print("=" * 60)

    result = validate_quantization(
        baseline_path=args.baseline,
        quantized_path=args.quantized,
        quantization=args.quantization,
        dtype=args.dtype,
        max_ppl_delta=args.max_ppl_delta,
        num_samples=args.num_samples,
        val_data_path=args.val_data,
        verbose=True,
    )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.output}")

    # 返回码: 0=通过, 1=失败/错误
    if result["error"]:
        print(f"\n[验证失败] {result['error']}")
        sys.exit(1)
    if not result["passed"]:
        print(f"\n[验证未通过] PPL 偏移 {result['delta']:.4f} 超过阈值 {result['threshold']}")
        print("建议检查校准数据质量或增加 num_samples")
        sys.exit(1)
    print(f"\n[验证通过] PPL 偏移 {result['delta']:.4f}，在阈值范围内")
    sys.exit(0)


if __name__ == "__main__":
    cli_validate()
