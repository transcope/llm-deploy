"""对比原模型 vs GPTQ 量化模型的领域精度

用 transformers 加载原模型, 用 gptqmodel + TORCH backend 加载量化模型,
在领域 Benchmark 数据上对比精度.
"""
import sys
import json
import time
import argparse
from pathlib import Path

sys.path.insert(0, "/volume/workspace/llm-deploy/llm_deploy")


def load_benchmark(path, num_samples=0, seed=42):
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                samples.append(json.loads(line))
            except Exception:
                continue
    if num_samples > 0 and len(samples) > num_samples:
        import random
        rng = random.Random(seed)
        samples = rng.sample(samples, num_samples)
    return samples


def score_keyword(model_answer, ground_truth):
    """关键词匹配评分: 核心关键词的召回率"""
    import re
    ans = model_answer.lower().strip()
    gt = ground_truth.lower().strip()
    if not gt:
        return 0.0
    # 提取关键词 (去掉停用词和标点)
    stopwords = set("的 了 是 在 和 与 及 或 一个 一种 这个 那个 我们 你们 他们 它 其 中 对 为 从 到 于 而 但 也 都 很 更 最 有 无 不 没 就 才 只 又 再 还 已 将 会 能 可 以 被 把 让 使 用 通过 进行 提供 实现 需要 可以 应该 可能 主要 相关 以及 等 等".split())
    # 提取 gt 中的关键词 (2字以上)
    words = re.findall(r"[\u4e00-\u9fa5]{2,}|[A-Za-z0-9_]+", gt)
    keywords = [w for w in words if w not in stopwords and len(w) >= 2]
    if not keywords:
        return 0.0
    matched = sum(1 for w in keywords if w in ans)
    return matched / len(keywords)


def score_exact(model_answer, ground_truth):
    """精确匹配评分"""
    return 1.0 if model_answer.strip().lower() == ground_truth.strip().lower() else 0.0


def score_answer(model_answer, ground_truth, scoring):
    if scoring == "exact_match":
        return score_exact(model_answer, ground_truth)
    return score_keyword(model_answer, ground_truth)


def load_transformers_model(model_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"加载 transformers 模型: {model_path}")
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, tok


def load_gptq_model(model_path):
    import torch
    from qwen3_gptq_adapter import install_qwen3_gptq_adapter
    from gptqmodel import GPTQModel
    from gptqmodel.utils.backend import BACKEND
    from transformers import AutoTokenizer
    print(f"加载 gptqmodel 模型: {model_path}")
    install_qwen3_gptq_adapter()
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = GPTQModel.from_quantized(
        model_path, device="cuda:0", backend=BACKEND.TORCH,
    )
    model.eval()
    return model, tok


def generate(model, tok, question, max_tokens=256):
    import torch
    messages = [
        {"role": "system", "content": "你是通信领域专家，请准确回答以下问题。"},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=4096)
    input_ids = inputs.input_ids.to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            input_ids, max_new_tokens=max_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tok.eos_token_id,
        )
    generated = outputs[0][input_ids.shape[1]:]
    return tok.decode(generated, skip_special_tokens=True).strip()


def evaluate(model, tok, samples, max_tokens=256):
    results = []
    correct = 0
    total = len(samples)
    start = time.time()
    for idx, sample in enumerate(samples):
        question = sample.get("question", "")
        answer_gt = sample.get("answer", "")
        source = sample.get("source", "unknown")
        scoring = sample.get("scoring", "keyword")
        if not question:
            continue
        try:
            model_answer = generate(model, tok, question, max_tokens)
        except Exception as e:
            model_answer = f"[error: {e}]"
        score = score_answer(model_answer, answer_gt, scoring)
        passed = score >= 0.35
        if passed:
            correct += 1
        results.append({
            "index": idx, "question": question, "ground_truth": answer_gt,
            "model_answer": model_answer, "score": round(score, 4),
            "scoring": scoring, "source": source, "passed": passed,
        })
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  [{idx+1}/{total}] source={source} score={score:.3f} {'PASS' if passed else 'FAIL'}")
    elapsed = time.time() - start
    return results, correct, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="原模型路径")
    parser.add_argument("--quantized", required=True, help="量化模型路径")
    parser.add_argument("--benchmark", default="/volume/workspace/llm-deploy/data/evaluation/accuracy_benchmark.jsonl")
    parser.add_argument("--num-samples", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--output", default="/volume/workspace/llm-deploy/results/compare")
    args = parser.parse_args()

    samples = load_benchmark(args.benchmark, args.num_samples)
    print(f"加载 {len(samples)} 条 Benchmark 数据")

    # 评测原模型
    print("\n===== 评测原模型 (FP16) =====")
    model_b, tok_b = load_transformers_model(args.baseline)
    res_b, correct_b, elapsed_b = evaluate(model_b, tok_b, samples, args.max_tokens)
    acc_b = correct_b / len(samples) if samples else 0
    print(f"原模型: 准确率 {acc_b:.4f} ({correct_b}/{len(samples)}), 耗时 {elapsed_b:.1f}s")
    del model_b
    import torch
    torch.cuda.empty_cache()

    # 评测量化模型
    print("\n===== 评测量化模型 (GPTQ) =====")
    model_q, tok_q = load_gptq_model(args.quantized)
    res_q, correct_q, elapsed_q = evaluate(model_q, tok_q, samples, args.max_tokens)
    acc_q = correct_q / len(samples) if samples else 0
    print(f"量化模型: 准确率 {acc_q:.4f} ({correct_q}/{len(samples)}), 耗时 {elapsed_q:.1f}s")

    # 输出报告
    report = {
        "meta": {
            "baseline": args.baseline, "quantized": args.quantized,
            "benchmark": args.benchmark, "num_samples": len(samples),
        },
        "baseline": {"accuracy": round(acc_b, 4), "correct": correct_b, "total": len(samples), "elapsed": round(elapsed_b, 1)},
        "quantized": {"accuracy": round(acc_q, 4), "correct": correct_q, "total": len(samples), "elapsed": round(elapsed_q, 1)},
        "delta": round(acc_q - acc_b, 4),
        "results": {"baseline": res_b, "quantized": res_q},
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "compare_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存: {out / 'compare_report.json'}")
    print(f"\n===== 对比结果 =====")
    print(f"原模型  准确率: {acc_b:.4f}")
    print(f"量化模型准确率: {acc_q:.4f}")
    print(f"精度差 (delta): {acc_q - acc_b:+.4f}")


if __name__ == "__main__":
    main()
