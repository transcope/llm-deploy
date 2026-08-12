#!/usr/bin/env python3
"""
领域精度评测脚本

在 domain-specific benchmark 数据集上评测模型精度。
两种运行模式:
  1. API 模式 (默认): 通过 OpenAI 兼容 API 评测量化/基线模型
  2. 本地模式 (--local): 直接加载模型评测 (需要 GPU)

用法:
    # 通过 API 评测 (服务启动后)
    python llm_deploy/benchmark_domain.py \
        --base-url http://192.168.192.186:8000 \
        --model Qwen3-8B-GPTQ \
        --output results/domain_eval.json

    # 评测基线模型 (通过 API)
    python llm_deploy/benchmark_domain.py \
        --base-url http://192.168.192.186:8001 \
        --model Mind-SLLM-Qwen3-8B \
        --output results/domain_baseline.json

    # 本地加载模型评测 (需 GPU)
    python llm_deploy/benchmark_domain.py \
        --local \
        --model /app/local_models/Mind-SLLM-Qwen3-8B \
        --output results/domain_eval.json

    # 本地评测 GPTQ 量化模型 (V100 + vLLM 0.8.5)
    python llm_deploy/benchmark_domain.py \
        --local \
        --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
        --quantization gptq \
        --output results/domain_gptq.json

    # 指定 benchmark 数据
    python llm_deploy/benchmark_domain.py \
        --benchmark data/custom_data/accuracy_benchmark.jsonl \
        --num-samples 50 \
        --base-url http://localhost:8000 \
        --model default
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None
    print("[警告] requests 未安装, API 模式不可用", file=sys.stderr)


# ============================================================
# 评分函数
# ============================================================

def _extract_keywords(text: str) -> set[str]:
    """从文本中提取关键词 (中文/英文/数字)"""
    # 中文词组
    cjk = set(re.findall(r'[\u4e00-\u9fff]{2,}', text))
    # 英文单词 (>= 3 字母, 去停用词)
    stopwords = {"the", "is", "are", "was", "were", "be", "been", "being",
                 "have", "has", "had", "do", "does", "did", "will", "would",
                 "can", "could", "may", "might", "shall", "should", "to",
                 "of", "in", "for", "on", "with", "at", "by", "from", "as",
                 "an", "a", "and", "or", "but", "if", "then", "else", "not",
                 "this", "that", "these", "those", "it", "its", "you"}
    eng_words = set()
    for word in re.findall(r'[a-zA-Z]{3,}', text.lower()):
        if word not in stopwords:
            eng_words.add(word)
    # 数值 (整数, 浮点, 百分数)
    nums = set(re.findall(r'\b\d+\.?\d*%?\b', text))

    keywords = cjk | eng_words | nums
    # 过滤掉单字符和纯标点
    return {k for k in keywords if len(k.strip()) > 1}


def score_exact_match(model_answer: str, ground_truth: str) -> dict:
    """精确匹配评分: 检查关键短语/数值的覆盖率

    返回:
        score: 0.0 ~ 1.0
        match_details: 匹配详情
    """
    ans_lower = model_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # 1. 数值匹配 (如果有数值)
    gt_nums = set(re.findall(r'\b\d+\.?\d*%?\b', gt_lower))
    ans_nums = set(re.findall(r'\b\d+\.?\d*%?\b', ans_lower))
    num_recall = 0.0
    if gt_nums:
        matched_nums = gt_nums & ans_nums
        num_recall = len(matched_nums) / len(gt_nums) if gt_nums else 0.0

    # 2. 关键词匹配
    gt_kw = _extract_keywords(gt_lower)
    ans_kw = _extract_keywords(ans_lower)
    kw_recall = 0.0
    if gt_kw:
        matched_kw = gt_kw & ans_kw
        kw_recall = len(matched_kw) / len(gt_kw) if gt_kw else 0.0

    # 3. combined score: 加权 (数值>关键词)
    if gt_nums:
        score = 0.6 * num_recall + 0.4 * kw_recall
    else:
        score = kw_recall

    return {
        "score": round(score, 4),
        "num_recall": round(num_recall, 4),
        "kw_recall": round(kw_recall, 4),
        "matched_keywords": list(matched_kw) if gt_kw else [],
        "total_keywords": len(gt_kw),
        "matched_nums": list(matched_nums) if gt_nums else [],
        "total_nums": len(gt_nums),
    }


def score_keyword_match(model_answer: str, ground_truth: str) -> dict:
    """关键词匹配评分: 核心关键词的召回率

    适用于长答案 (讲解/代码/解题过程)
    """
    ans_lower = model_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()

    gt_kw = _extract_keywords(gt_lower)
    ans_kw = _extract_keywords(ans_lower)

    if not gt_kw:
        return {"score": 1.0, "kw_recall": 1.0,
                "matched_keywords": [], "total_keywords": 0}

    matched_kw = gt_kw & ans_kw
    recall = len(matched_kw) / len(gt_kw)

    return {
        "score": round(recall, 4),
        "kw_recall": round(recall, 4),
        "matched_keywords": list(matched_kw),
        "total_keywords": len(gt_kw),
        "matched_count": len(matched_kw),
    }


def score_answer(model_answer: str, ground_truth: str,
                 scoring: str = "keyword") -> dict:
    """统一评分入口"""
    if scoring == "exact_match":
        return score_exact_match(model_answer, ground_truth)
    else:  # keyword
        return score_keyword_match(model_answer, ground_truth)


# ============================================================
# API 调用
# ============================================================

def call_api(base_url: str, model: str, messages: list,
             max_tokens: int = 1024, temperature: float = 0.0,
             timeout: int = 120) -> str | None:
    """通过 OpenAI 兼容 API 调用模型"""
    if requests is None:
        raise RuntimeError("requests 未安装, 无法使用 API 模式")

    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        if resp.status_code != 200:
            print(f"  [API错误] status={resp.status_code} {resp.text[:200]}",
                  file=sys.stderr)
            return None
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return None
        return choices[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"  [API异常] {e}", file=sys.stderr)
        return None


# ============================================================
# Benchmark 加载
# ============================================================

def load_benchmark(path: str, num_samples: int = 0, seed: int = 42) -> list[dict]:
    """加载 benchmark JSONL, 可选采样"""
    path = Path(path)
    if not path.exists():
        print(f"[错误] Benchmark 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not samples:
        print(f"[错误] Benchmark 文件为空或格式错误: {path}", file=sys.stderr)
        sys.exit(1)

    if 0 < num_samples < len(samples):
        random.seed(seed)
        samples = random.sample(samples, num_samples)

    return samples


# ============================================================
# 主流程
# ============================================================

def evaluate_api(args):
    """API 模式: 通过部署的服务评测"""
    print("=" * 60)
    print("领域精度评测 (API 模式)")
    print(f"  API:     {args.base_url}")
    print(f"  Model:   {args.model}")
    print(f"  Dataset: {args.benchmark} ({args.num_samples} samples)")
    print("=" * 60)

    samples = load_benchmark(args.benchmark, args.num_samples)
    print(f"\n加载 {len(samples)} 条 benchmark 样本\n")

    results = []
    correct_count = 0
    total_count = len(samples)
    per_source = {}

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        answer_gt = sample.get("answer", "")
        source = sample.get("source", "unknown")
        scoring = sample.get("scoring", "keyword")

        if not question:
            continue

        # 构建对话
        messages = [
            {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"},
            {"role": "user", "content": question},
        ]

        # 调用 API
        model_answer = call_api(
            args.base_url, args.model, messages,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            timeout=args.timeout,
        )

        if model_answer is None:
            result = {
                "index": idx,
                "question": question,
                "ground_truth": answer_gt,
                "model_answer": None,
                "score": 0.0,
                "scoring": scoring,
                "source": source,
                "error": "API call failed",
            }
        else:
            # 评分
            score_info = score_answer(model_answer, answer_gt, scoring)
            passed = score_info["score"] >= args.pass_threshold
            result = {
                "index": idx,
                "question": question,
                "ground_truth": answer_gt,
                "model_answer": model_answer,
                "score": score_info["score"],
                "scoring": scoring,
                "source": source,
                "passed": passed,
                "details": score_info,
            }
            if passed:
                correct_count += 1

        results.append(result)

        # 按来源统计
        if source not in per_source:
            per_source[source] = {"total": 0, "correct": 0, "score_sum": 0.0}
        per_source[source]["total"] += 1
        per_source[source]["score_sum"] += result.get("score", 0.0)
        if result.get("passed", False):
            per_source[source]["correct"] += 1

        # 进度
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{total_count}] source={source} score={result.get('score', 0):.3f} "
                  f"{'✓' if result.get('passed', False) else '✗'}")

        # 避免请求过快
        if args.delay > 0:
            time.sleep(args.delay)

    # ---- 汇总报告 ----
    overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
    overall_avg_score = sum(r.get("score", 0.0) for r in results) / total_count if total_count > 0 else 0.0

    report = {
        "meta": {
            "mode": "api",
            "base_url": args.base_url,
            "model": args.model,
            "benchmark": str(args.benchmark),
            "num_samples": total_count,
            "pass_threshold": args.pass_threshold,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall": {
            "accuracy": round(overall_accuracy, 4),
            "avg_score": round(overall_avg_score, 4),
            "correct": correct_count,
            "total": total_count,
        },
        "per_source": {},
        "results": results,
    }

    for src, stats in sorted(per_source.items()):
        avg = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0.0
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        report["per_source"][src] = {
            "accuracy": round(acc, 4),
            "avg_score": round(avg, 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    # 打印汇总
    print(f"\n{'=' * 60}")
    print(f"评测完成")
    print(f"  总体准确率: {overall_accuracy:.2%} ({correct_count}/{total_count})")
    print(f"  平均得分:   {overall_avg_score:.4f}")
    print(f"  阈值:       pass >= {args.pass_threshold}")
    print(f"\n  按来源:")
    for src, stats in sorted(per_source.items()):
        avg = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0.0
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"    {src:20s}: acc={acc:.2%}  avg_score={avg:.4f}  "
              f"({stats['correct']}/{stats['total']})")

    return report


def evaluate_local_vllm(args, samples):
    """本地模式 (vLLM 后端): 使用 vLLM 批量推理

    V100 专用参数 (vLLM 0.8.5):
      - VLLM_ATTENTION_BACKEND=XFORMERS (V100 不支持 Flash Attention)
      - enforce_eager=True (避免 CUDA graph 问题)
      - dtype=float16 (V100 不支持 bfloat16)
      - quantization=gptq (量化模型)
    """
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[错误] vLLM 未安装", file=sys.stderr)
        sys.exit(1)

    # V100 需要 XFORMERS attention backend
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "XFORMERS")

    # 构建 prompts
    prompts = []
    for sample in samples:
        question = sample.get("question", "")
        messages = [
            {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"},
            {"role": "user", "content": question},
        ]
        prompts.append(messages)

    print(f"加载模型: {args.model}")
    print(f"  tensor_parallel_size={args.tp}, quantization={args.quantization}")
    print(f"  enforce_eager={args.enforce_eager}, gpu_util={args.gpu_util}, max_model_len={args.max_model_len}")
    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        dtype="float16",
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
    )
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # vLLM 0.8.5 V0 引擎 bug: 单条/无 system 消息的 prompt 会产生退化输出 "!!!!"
    # 用一条带 system 消息的不同 dummy prompt 填充到 2 条, 取第一条结果
    sys_msg = {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"}
    dummy = [sys_msg, {"role": "user", "content": "请介绍一下通信中的调制方式。"}]
    batch_prompts = prompts + [dummy]

    print(f"\n开始评测 ({len(prompts)} 条)...")
    start_time = time.time()
    outputs = llm.chat(batch_prompts, sampling_params)
    # 丢弃 dummy 的最后一条结果
    outputs = outputs[:len(prompts)]
    elapsed = time.time() - start_time
    print(f"推理完成, 耗时 {elapsed:.1f}s ({len(prompts)/elapsed:.1f} samples/s)")

    # 评分
    results = []
    correct_count = 0
    total_count = len(samples)
    per_source = {}

    for idx, (sample, output) in enumerate(zip(samples, outputs)):
        question = sample.get("question", "")
        answer_gt = sample.get("answer", "")
        source = sample.get("source", "unknown")
        scoring = sample.get("scoring", "keyword")

        model_answer = output.outputs[0].text.strip() if output.outputs else ""

        score_info = score_answer(model_answer, answer_gt, scoring)
        passed = score_info["score"] >= args.pass_threshold

        result = {
            "index": idx,
            "question": question,
            "ground_truth": answer_gt,
            "model_answer": model_answer,
            "score": score_info["score"],
            "scoring": scoring,
            "source": source,
            "passed": passed,
            "details": score_info,
        }
        results.append(result)

        if passed:
            correct_count += 1

        if source not in per_source:
            per_source[source] = {"total": 0, "correct": 0, "score_sum": 0.0}
        per_source[source]["total"] += 1
        per_source[source]["score_sum"] += score_info["score"]
        if passed:
            per_source[source]["correct"] += 1

        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{total_count}] source={source} score={score_info['score']:.3f} "
                  f"{'✓' if passed else '✗'}")

    return results, correct_count, elapsed, per_source


def evaluate_local_transformers(args, samples):
    """本地模式 (Transformers 后端): 使用 HuggingFace Transformers 逐条推理

    适用于 vLLM 不支持的配置 (如 V100 + Qwen3 的 LLVM 错误),
    以及量化模型的精度验证.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[错误] transformers/pytorch 未安装", file=sys.stderr)
        sys.exit(1)

    import warnings
    warnings.filterwarnings("ignore")

    print(f"加载模型: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,  # single-GPU, no accelerate overhead
        trust_remote_code=True,
    )
    model.eval()
    model = model.to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 抑制 Qwen3 thinking 模式
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    model.generation_config.do_sample = False

    print(f"  Device: {model.device}, Params: {model.num_parameters()/1e9:.2f}B")

    # 推理
    results = []
    correct_count = 0
    total_count = len(samples)
    per_source = {}

    print(f"\n开始评测 ({total_count} 条)...")
    start_time = time.time()

    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        answer_gt = sample.get("answer", "")
        source = sample.get("source", "unknown")
        scoring = sample.get("scoring", "keyword")

        messages = [
            {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"},
            {"role": "user", "content": question},
        ]
        # suppress thinking mode for direct answers
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs.input_ids.to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids = outputs[0][input_ids.shape[1]:]
        model_answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not model_answer:
            model_answer = "[empty]"

        score_info = score_answer(model_answer, answer_gt, scoring)
        passed = score_info["score"] >= args.pass_threshold

        result = {
            "index": idx,
            "question": question,
            "ground_truth": answer_gt,
            "model_answer": model_answer,
            "score": score_info["score"],
            "scoring": scoring,
            "source": source,
            "passed": passed,
            "details": score_info,
        }
        results.append(result)

        if passed:
            correct_count += 1

        if source not in per_source:
            per_source[source] = {"total": 0, "correct": 0, "score_sum": 0.0}
        per_source[source]["total"] += 1
        per_source[source]["score_sum"] += score_info["score"]
        if passed:
            per_source[source]["correct"] += 1

        if (idx + 1) % 5 == 0 or idx == 0:
            t = time.time() - start_time
            rate = (idx + 1) / t if t > 0 else 0
            print(f"  [{idx+1}/{total_count}] {source:10s} score={score_info['score']:.4f} "
                  f"{'✓' if passed else '✗'}  ({t:.0f}s, {rate:.2f} samp/s)")

    elapsed = time.time() - start_time
    return results, correct_count, elapsed, per_source


def evaluate_local(args):
    """本地模式: 直接加载模型评测 (需要 GPU)"""
    print("=" * 60)
    print("领域精度评测 (本地模式)")
    print(f"  Model:   {args.model}")
    print(f"  Backend: {args.backend}")
    print(f"  Dataset: {args.benchmark} ({args.num_samples} samples)")
    print("=" * 60)

    samples = load_benchmark(args.benchmark, args.num_samples)
    print(f"\n加载 {len(samples)} 条 benchmark 样本\n")

    # 选择后端
    if args.backend == "transformers":
        results, correct_count, elapsed, per_source = evaluate_local_transformers(args, samples)
    else:
        results, correct_count, elapsed, per_source = evaluate_local_vllm(args, samples)

    total_count = len(samples)
    overall_accuracy = correct_count / total_count if total_count > 0 else 0.0
    overall_avg_score = sum(r.get("score", 0.0) for r in results) / total_count if total_count > 0 else 0.0

    report = {
        "meta": {
            "mode": f"local-{args.backend}",
            "model": args.model,
            "backend": args.backend,
            "benchmark": str(args.benchmark),
            "num_samples": total_count,
            "pass_threshold": args.pass_threshold,
            "enable_thinking": not args.no_thinking,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall": {
            "accuracy": round(overall_accuracy, 4),
            "avg_score": round(overall_avg_score, 4),
            "correct": correct_count,
            "total": total_count,
        },
        "per_source": {},
        "results": results,
    }

    for src, stats in sorted(per_source.items()):
        avg = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0.0
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        report["per_source"][src] = {
            "accuracy": round(acc, 4),
            "avg_score": round(avg, 4),
            "correct": stats["correct"],
            "total": stats["total"],
        }

    print(f"\n{'=' * 60}")
    print(f"评测完成")
    print(f"  总体准确率: {overall_accuracy:.2%} ({correct_count}/{total_count})")
    print(f"  平均得分:   {overall_avg_score:.4f}")
    print(f"  阈值:       pass >= {args.pass_threshold}")
    print(f"\n  按来源:")
    for src, stats in sorted(per_source.items()):
        avg = stats["score_sum"] / stats["total"] if stats["total"] > 0 else 0.0
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        print(f"    {src:20s}: acc={acc:.2%}  avg_score={avg:.4f}  "
              f"({stats['correct']}/{stats['total']})")

    return report


def main():
    parser = argparse.ArgumentParser(
        description="领域精度评测工具 (domain-specific benchmark)"
    )
    # 数据源
    parser.add_argument(
        "--benchmark", "-b",
        default=str(Path(__file__).resolve().parent.parent
                    / "data" / "custom_data" / "accuracy_benchmark.jsonl"),
        help="Benchmark JSONL 路径"
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=0,
        help="采样数 (0=全部)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="采样随机种子"
    )

    # 模型连接
    parser.add_argument(
        "--base-url", type=str, default=None,
        help="OpenAI 兼容 API 地址 (e.g. http://localhost:8000)"
    )
    parser.add_argument(
        "--model", type=str, default="default",
        help="API mode: model name; Local mode: model path"
    )
    parser.add_argument(
        "--local", action="store_true",
        help="本地加载模型 (需 GPU, 使用 vLLM 或 Transformers)"
    )
    parser.add_argument(
        "--backend", type=str, default="vllm", choices=["vllm", "transformers"],
        help="本地模式推理后端 (默认 vllm; V100+Qwen3 用 vllm 0.8.5 或 transformers)"
    )
    parser.add_argument(
        "--tp", type=int, default=1,
        help="本地模式: tensor_parallel_size (仅 vllm 后端)"
    )
    parser.add_argument(
        "--quantization", type=str, default=None,
        help="本地模式 (vllm 后端): 量化方式 (gptq)"
    )
    parser.add_argument(
        "--enforce-eager", action="store_true", default=True,
        help="本地模式 (vllm 后端): 禁用 CUDA graph (V100 需要)"
    )
    parser.add_argument(
        "--gpu-util", type=float, default=0.9,
        help="本地模式 (vllm 后端): GPU 内存利用率"
    )
    parser.add_argument(
        "--max-model-len", type=int, default=4096,
        help="本地模式 (vllm 后端): 最大序列长度"
    )
    parser.add_argument(
        "--no-thinking", action="store_true", default=True,
        help="禁用 Qwen3 thinking 模式 (仅 transformers 后端)"
    )

    # 生成参数
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="最大生成长度 (transformers 后端默认 256; vllm/API 默认 1024)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="生成温度 (0 = 贪婪)"
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="API 超时 (秒)"
    )
    parser.add_argument(
        "--delay", type=float, default=0.0,
        help="API 请求间隔 (秒)"
    )

    # 评分
    parser.add_argument(
        "--pass-threshold", type=float, default=0.35,
        help="通过阈值 (默认 0.35, 即关键词召回率 >= 35% 视为正确)"
    )

    # 输出
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="结果输出 JSON 文件"
    )

    args = parser.parse_args()

    # 参数校验
    if args.local and args.base_url:
        print("[错误] --local 和 --base-url 不能同时使用", file=sys.stderr)
        sys.exit(1)

    if not args.local and not args.base_url:
        # 默认: 尝试本地 localhost:8000
        args.base_url = "http://localhost:8000"
        print(f"[信息] 未指定连接方式, 默认使用 API: {args.base_url}", file=sys.stderr)
        print(f"       用 --local 切换为本地加载, 或指定 --base-url", file=sys.stderr)

    # 评测
    if args.local:
        report = evaluate_local(args)
    else:
        report = evaluate_api(args)

    # 输出
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
