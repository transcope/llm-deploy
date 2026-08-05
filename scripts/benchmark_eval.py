#!/usr/bin/env python3
"""
大模型评测脚本 - 基于 lm-evaluation-harness 和 vLLM
支持精度评测(Accuracy)、困惑度(Perplexity)、速度评测(Throughput/Latency)

用法:
    # 基础精度评测
    python benchmark_eval.py --model Qwen/Qwen2.5-7B-Instruct --tasks gsm8k,hellaswag --output results/

    # 量化模型对比评测
    python benchmark_eval.py --model ./models/Qwen2.5-7B-AWQ --quantization awq --tasks gsm8k,humaneval --baseline-model Qwen/Qwen2.5-7B-Instruct

    # 性能基准测试 (吞吐/延迟)
    python benchmark_eval.py --model Qwen/Qwen2.5-7B-Instruct --perf-test --num-prompts 100 --max-tokens 512

    # 完整评测套件
    python benchmark_eval.py --model Qwen/Qwen2.5-7B-Instruct --suite full --output results/
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests


# 评测任务配置
BENCHMARK_TASKS = {
    "math": ["gsm8k", "mathqa"],
    "code": ["humaneval", "mbpp"],
    "reasoning": ["hellaswag", "winogrande", "arc_challenge"],
    "knowledge": ["mmlu", "truthfulqa_mc"],
    "perplexity": ["wikitext"],
    "chinese": ["ceval-valid", "cmmlu"],
}

FULL_SUITE = []
for tasks in BENCHMARK_TASKS.values():
    FULL_SUITE.extend(tasks)


def build_model_args(
    model_path: str,
    quantization: str = "",
    max_model_len: int = 4096,
    gpu_memory_utilization: float = 0.8,
    dtype: str = "auto",
    add_bos_token: bool = True,
    tensor_parallel: int = 1,
    enforce_eager: bool = False,
    max_num_seqs: Optional[int] = None,
) -> str:
    """构建 lm-eval 的 model_args 字符串

    注意: V100 (SM 7.0) 不支持 bfloat16, 请传入 dtype="float16"

    V100 32GB 显存调参建议 (8B Qwen3, vocab=151k):
      - enforce_eager=True: 跳过 CUDA graph 捕获, 避免捕获 35 个 shape 占满显存
        导致 sampler 的 log_softmax(151k vocab) 分配 2.3GB 失败而 OOM
      - gpu_memory_utilization<=0.6: 给 sampler 留出 logits 工作显存
      - max_num_seqs<=64: 限制并发序列数, 控制 logits 张量 [B, T, 151k] 大小
    """
    args = {
        "pretrained": model_path,
        "dtype": dtype,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": max_model_len,
        "add_bos_token": str(add_bos_token).lower(),
    }
    if quantization:
        args["quantization"] = quantization
    if tensor_parallel > 1:
        args["tensor_parallel_size"] = tensor_parallel
    if enforce_eager:
        args["enforce_eager"] = "true"
    if max_num_seqs is not None:
        args["max_num_seqs"] = max_num_seqs

    return ",".join(f"{k}={v}" for k, v in args.items())


def run_lm_eval(
    model_path: str,
    tasks: List[str],
    output_dir: str,
    quantization: str = "",
    num_fewshot: Optional[int] = None,
    batch_size: str = "auto",
    device: str = "cuda",
    max_model_len: int = 4096,
    limit: Optional[int] = None,
    dtype: str = "auto",
    tensor_parallel: int = 1,
    gpu_memory_utilization: float = 0.8,
    enforce_eager: bool = False,
    max_num_seqs: Optional[int] = None,
) -> Dict:
    """使用 lm-evaluation-harness 运行精度评测"""
    print(f"\n{'='*60}")
    print(f"运行 lm-eval: {model_path}")
    print(f"任务: {', '.join(tasks)}")
    print(f"{'='*60}")

    model_args = build_model_args(
        model_path=model_path,
        quantization=quantization,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        add_bos_token=True,
        tensor_parallel=tensor_parallel,
        enforce_eager=enforce_eager,
        max_num_seqs=max_num_seqs,
    )

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "vllm",
        "--model_args", model_args,
        "--tasks", ",".join(tasks),
        "--batch_size", batch_size,
        "--device", device,
        "--output_path", output_dir,
        "--log_samples",
    ]

    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"命令: {' '.join(cmd)}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        print(f"[错误] lm-eval 执行失败:\n{result.stderr}")
        return {"error": result.stderr, "elapsed": elapsed}

    print(f"[完成] 耗时: {elapsed:.1f}s")

    results = {"elapsed": elapsed, "raw_output": result.stdout}

    try:
        result_files = list(Path(output_dir).rglob("*.json"))
        if result_files:
            # 优先读取最新的结果文件
            result_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            with open(result_files[0], "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            results["scores"] = extract_scores(eval_results)
    except Exception as e:
        print(f"[警告] 无法解析结果文件: {e}")

    return results


def extract_scores(eval_results: Dict) -> Dict[str, float]:
    """从 lm-eval 结果中提取分数"""
    scores = {}
    try:
        results_data = eval_results.get("results", {})
        for task_name, task_results in results_data.items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)) and not metric.startswith("_"):
                    scores[f"{task_name}_{metric}"] = value
    except Exception:
        pass
    return scores


def send_stream_request(base_url: str, prompt: str, model_name: str, max_tokens: int) -> Dict:
    """向 vLLM 服务发送流式请求并测量 TTFT"""
    start = time.time()
    first_token_time = None
    completion_text = ""
    usage = {}

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            timeout=120,
            stream=True,
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue
            text = line.decode("utf-8", errors="ignore")
            if first_token_time is None:
                first_token_time = time.time() - start

            if not text.startswith("data: "):
                continue
            data = text[6:]
            if data.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage.update(chunk["usage"])
                delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                completion_text += delta
            except json.JSONDecodeError:
                continue

        elapsed = time.time() - start
        completion_tokens = usage.get("completion_tokens") or max(1, len(completion_text.split()))
        return {
            "latency": elapsed,
            "ttft": first_token_time if first_token_time is not None else elapsed,
            "tokens": completion_tokens,
            "tps": completion_tokens / elapsed if elapsed > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e)}


def run_perf_test(
    model_path: str,
    base_url: str = "http://localhost:8000",
    num_prompts: int = 100,
    max_tokens: int = 256,
    concurrency: int = 10,
) -> Dict:
    """性能基准测试 - 测试吞吐量和延迟，使用流式请求测量 TTFT"""
    print(f"\n{'='*60}")
    print(f"性能基准测试")
    print(f"服务地址: {base_url}")
    print(f"请求数: {num_prompts}, 最大token: {max_tokens}, 并发: {concurrency}")
    print(f"{'='*60}")

    test_prompts = [
        "请解释一下量子计算的基本原理。",
        "Write a Python function to calculate Fibonacci numbers.",
        "What are the main differences between TCP and UDP?",
        "请描述一下深度学习中的反向传播算法。",
        "Explain the concept of recursion in programming.",
    ] * (num_prompts // 5 + 1)
    test_prompts = test_prompts[:num_prompts]

    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=10)
        model_name = resp.json()["data"][0]["id"] if resp.status_code == 200 else "unknown"
    except Exception:
        model_name = "unknown"

    latencies = []
    ttfts = []
    tokens_generated = []

    start_all = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(send_stream_request, base_url, p, model_name, max_tokens)
            for p in test_prompts
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if "error" not in result:
                latencies.append(result["latency"])
                ttfts.append(result["ttft"])
                tokens_generated.append(result["tokens"])

    total_time = time.time() - start_all
    total_tokens = sum(tokens_generated)

    perf_results = {
        "total_requests": len(test_prompts),
        "successful_requests": len(latencies),
        "total_time_seconds": total_time,
        "total_tokens_generated": total_tokens,
        "throughput_requests_per_sec": len(latencies) / total_time if total_time > 0 else 0,
        "throughput_tokens_per_sec": total_tokens / total_time if total_time > 0 else 0,
        "latency_avg_seconds": statistics.mean(latencies) if latencies else 0,
        "latency_p50_seconds": statistics.median(latencies) if latencies else 0,
        "latency_p99_seconds": sorted(latencies)[int(len(latencies) * 0.99)] if len(latencies) > 1 else 0,
        "ttft_avg_seconds": statistics.mean(ttfts) if ttfts else 0,
        "ttft_p50_seconds": statistics.median(ttfts) if ttfts else 0,
        "ttft_p99_seconds": sorted(ttfts)[int(len(ttfts) * 0.99)] if len(ttfts) > 1 else 0,
        "tokens_per_request_avg": statistics.mean(tokens_generated) if tokens_generated else 0,
    }

    print(f"\n性能测试结果:")
    print(f"  总请求数: {perf_results['total_requests']}")
    print(f"  成功请求: {perf_results['successful_requests']}")
    print(f"  总耗时: {perf_results['total_time_seconds']:.2f}s")
    print(f"  总生成Token: {perf_results['total_tokens_generated']}")
    print(f"  吞吐 (req/s): {perf_results['throughput_requests_per_sec']:.2f}")
    print(f"  吞吐 (tok/s): {perf_results['throughput_tokens_per_sec']:.2f}")
    print(f"  平均延迟: {perf_results['latency_avg_seconds']:.3f}s")
    print(f"  P50延迟: {perf_results['latency_p50_seconds']:.3f}s")
    print(f"  P99延迟: {perf_results['latency_p99_seconds']:.3f}s")
    print(f"  平均TTFT: {perf_results['ttft_avg_seconds']:.3f}s")
    print(f"  P99 TTFT: {perf_results['ttft_p99_seconds']:.3f}s")

    return perf_results


def run_quantization_comparison(
    baseline_model: str,
    quantized_models: Dict[str, str],
    tasks: List[str],
    output_dir: str,
) -> Dict:
    """量化模型对比评测 - 对比不同量化方案的精度损失"""
    print(f"\n{'='*70}")
    print("量化模型对比评测")
    print(f"基线模型: {baseline_model}")
    print(f"对比模型: {list(quantized_models.keys())}")
    print(f"{'='*70}")

    results = {}

    print(f"\n[1/{len(quantized_models) + 1}] 评测基线模型 (FP16/BF16)...")
    baseline_results = run_lm_eval(
        baseline_model, tasks, os.path.join(output_dir, "baseline")
    )
    results["baseline"] = baseline_results

    for i, (name, model_path) in enumerate(quantized_models.items(), 2):
        print(f"\n[{i}/{len(quantized_models)+1}] 评测 {name}...")
        # 名称为已知量化方法时才传给 vLLM, 否则由模型配置自动识别
        known_methods = {"awq", "gptq", "fp8", "marlin", "bitsandbytes", "compressed-tensors"}
        quant_results = run_lm_eval(
            model_path, tasks, os.path.join(output_dir, name),
            quantization=name if name in known_methods else "",
        )
        results[name] = quant_results

    print(f"\n{'='*70}")
    print("精度损失对比:")
    print(f"{'='*70}")

    baseline_scores = results.get("baseline", {}).get("scores", {})
    for name, result in results.items():
        if name == "baseline":
            continue
        quant_scores = result.get("scores", {})
        print(f"\n{name}:")
        for metric, baseline_val in baseline_scores.items():
            quant_val = quant_scores.get(metric, 0)
            if baseline_val > 0:
                loss = (baseline_val - quant_val) / baseline_val * 100
                flag = "🟢" if abs(loss) <= 1 else "🟡" if abs(loss) <= 3 else "🟠" if abs(loss) <= 5 else "🔴"
                print(f"  {metric}: 基线={baseline_val:.4f}, 量化={quant_val:.4f}, 损失={loss:.2f}% {flag}")

    return results


def main():
    parser = argparse.ArgumentParser(description="大模型评测工具")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径或 HuggingFace 模型 ID")
    parser.add_argument("--tasks", type=str, default="gsm8k,hellaswag",
                        help="评测任务，逗号分隔 (如: gsm8k,hellaswag,humaneval)")
    parser.add_argument("--suite", type=str, default="",
                        choices=["", "math", "code", "reasoning", "knowledge", "full"],
                        help="评测套件")
    parser.add_argument("--output", type=str, default="./results",
                        help="结果输出目录")
    parser.add_argument("--quantization", type=str, default="",
                        help="量化类型 (用于 vLLM 加载)")
    parser.add_argument("--baseline-model", type=str, default="",
                        help="基线模型路径 (用于对比评测)")
    parser.add_argument("--perf-test", action="store_true",
                        help="运行性能基准测试 (需要 vLLM 服务已启动)")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000",
                        help="vLLM 服务地址")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="性能测试请求数")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="性能测试最大生成token数")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="性能测试并发数")
    parser.add_argument("--limit", type=int, default=None,
                        help="限制评测样本数 (用于快速测试)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="模型数据类型 (V100 不支持 bfloat16, 请使用 float16)")
    parser.add_argument("--tensor-parallel", type=int, default=1,
                        help="张量并行大小 (多卡评测大模型时使用)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="vLLM gpu_memory_utilization (V100 8B 建议 <=0.6, 给 sampler 留 logits 工作显存)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="禁用 CUDA graph 捕获 (V100 OOM 时开启, 跳过 35 shape 捕获释放数 GB 显存)")
    parser.add_argument("--max-num-seqs", type=int, default=None,
                        help="vLLM 最大并发序列数 (限制 logits [B,T,151k] 张量大小, V100 建议 <=64)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                        help="vLLM 最大上下文长度 (loglikelihood 任务上下文短, 降到 2048 可减半 KV cache "
                             "给 sampler get_logprobs([B,T,151k]) 腾出工作显存)")
    parser.add_argument("--skip-accuracy", action="store_true",
                        help="跳过精度评测 (配合 --perf-test 使用, 避免与已启动的服务争抢显存)")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if args.suite:
        tasks = BENCHMARK_TASKS.get(args.suite, FULL_SUITE if args.suite == "full" else [])
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    print("=" * 70)
    print("大模型评测工具")
    print("=" * 70)
    print(f"模型: {args.model}")
    print(f"任务: {tasks}")
    print(f"输出: {args.output}")
    print("=" * 70)

    all_results = {}

    if args.skip_accuracy and not args.perf_test:
        parser.error("--skip-accuracy 需要与 --perf-test 一起使用")

    if not args.skip_accuracy:
        print("\n[阶段 1/2] 精度评测...")
        eval_results = run_lm_eval(
            model_path=args.model,
            tasks=tasks,
            output_dir=args.output,
            quantization=args.quantization,
            limit=args.limit,
            dtype=args.dtype,
            tensor_parallel=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enforce_eager=args.enforce_eager,
            max_num_seqs=args.max_num_seqs,
            max_model_len=args.max_model_len,
        )
        all_results["accuracy"] = eval_results

        # lm-eval 失败时 (如 vLLM OOM) 必须非零退出, 否则 shell 的 set -e/pipefail
        # 抓不到, stage 2 会在 stage 1 失败后继续跑 (浪费 50 分钟重跑同一个 OOM).
        if isinstance(eval_results, dict) and "error" in eval_results:
            # 仍写出 benchmark_results.json (含 traceback 供诊断), 再退出
            result_file = os.path.join(args.output, "benchmark_results.json")
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n[失败] lm-eval 报错, 结果已存到 {result_file}, 退出码 1", file=sys.stderr)
            sys.exit(1)

    if args.perf_test:
        print("\n[阶段 2/2] 性能基准测试...")
        perf_results = run_perf_test(
            model_path=args.model,
            base_url=args.base_url,
            num_prompts=args.num_prompts,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
        )
        all_results["performance"] = perf_results

    if args.baseline_model:
        print("\n[对比评测] 量化模型 vs 基线模型...")
        comparison_results = run_quantization_comparison(
            baseline_model=args.baseline_model,
            quantized_models={args.quantization or "quantized": args.model},
            tasks=tasks[:2],
            output_dir=args.output,
        )
        all_results["comparison"] = comparison_results

    result_file = os.path.join(args.output, "benchmark_results.json")
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'='*70}")
    print(f"评测完成! 结果已保存到: {result_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
