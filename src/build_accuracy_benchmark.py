#!/usr/bin/env python3
"""
领域精度评测 Benchmark 构建工具

从 data/custom_data/ 下的多源领域数据中提取 question-answer 对，
构建可用于 domain-specific 精度评测的 Benchmark 数据集。

与 build_calibration_data.py 的区别:
  - build_calibration_data.py: 输出 messages 格式, 用于量化校准
  - build_accuracy_benchmark.py: 输出 question-answer 对, 用于精度评测

输出格式: JSONL, 每行 {"question": "...", "answer": "...", "source": "..."}

用法:
    # 默认: 自动发现所有数据源, 构建 200 条
    python src/build_accuracy_benchmark.py

    # 指定样本数
    python src/build_accuracy_benchmark.py --num-samples 500

    # 列出可用数据源
    python src/build_accuracy_benchmark.py --list-sources
"""

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path


CUSTOM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "custom_data"


# ============================================================
# 1. QA 提取函数: 每种格式提取 question + answer + 评分辅助信息
# ============================================================

def extract_alpaca_qa(obj: dict) -> dict | None:
    """Alpaca 格式: instruction/input/output → QA pair

    评分策略: 答案通常是文本讲解, 用 keyword_match 评分
    """
    inst = (obj.get("instruction") or "").strip()
    inp = (obj.get("input") or "").strip()
    out = (obj.get("output") or "").strip()
    if not inst or not out:
        return None
    question = inst
    if inp:
        question += "\n" + inp
    return {
        "question": question,
        "answer": out,
        "source": "alpaca",
        "scoring": "keyword",  # 关键词匹配
    }


def extract_messages_qa(obj: dict) -> dict | None:
    """messages 格式: 取第一轮 user/assistant 作为 QA pair

    评分策略: assistant 回答通常是文本讲解, 用 keyword_match 评分
    """
    msgs = obj.get("messages")
    if not msgs or not isinstance(msgs, list):
        return None
    # 找第一组 user + assistant
    user_text = ""
    assistant_text = ""
    for m in msgs:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text and user_text:
            assistant_text = content
            break
    if not user_text or not assistant_text:
        return None
    return {
        "question": user_text,
        "answer": assistant_text,
        "source": "messages",
        "scoring": "keyword",
    }


def extract_tasks_qa(obj: dict) -> list[dict]:
    """tasks 格式: question + ground_truth → QA pair

    评分策略: ground_truth 可能是确定性答案, 支持 exact_match 或 keyword_match
    """
    results = []
    tasks = obj.get("tasks", [])
    for t in tasks:
        q = (t.get("question") or "").strip()
        gt = (t.get("ground_truth") or "").strip()
        desc = (t.get("task_description") or "").strip()
        if not q:
            continue
        # 优先用 ground_truth (更精确), 没有则用 task_description
        answer = gt if gt else desc
        if not answer:
            continue
        # 判断答案类型: 短答案用 exact_match, 长答案用 keyword
        scoring = "exact_match" if len(answer) < 100 else "keyword"
        results.append({
            "question": q,
            "answer": answer,
            "source": "tasks",
            "scoring": scoring,
        })
    return results


def extract_codegen_qa(obj: dict) -> dict | None:
    """codegen 格式: question + code → QA pair

    评分策略: 代码生成题, 用 keyword_match 检查关键函数名/逻辑
    """
    q = (obj.get("question") or "").strip()
    code = (obj.get("code") or "").strip()
    if not q or not code:
        return None
    return {
        "question": q,
        "answer": code,
        "source": "codegen",
        "scoring": "keyword",
    }


def extract_math_qa(obj: dict) -> dict | None:
    """math 格式: 解题过程 + 答案 → QA pair

    评分策略: 答案通常在 \boxed{} 中, 支持 exact_match 提取数值答案
    """
    cat = (obj.get("category") or ["未知"])[0] if obj.get("category") else "未知"
    explanation = (obj.get("explanation") or "").strip()
    answer_new = obj.get("answer_new") or []
    answer_post = (obj.get("answer_post") or "").strip()

    if not explanation:
        return None

    # 从 answer_new/answer_post 提取答案
    answer_text = ""
    for a in answer_new:
        if isinstance(a, str) and a.strip():
            # 提取 \boxed{...} 中的内容作为精确答案
            m = re.search(r'\\boxed\{([^}]+)\}', a)
            if m:
                answer_text += m.group(1).strip() + "\n"
            else:
                answer_text += a.strip() + "\n"
    if not answer_text and answer_post:
        answer_text = answer_post

    question = f"请解答这道{cat}题，给出详细的解题步骤和最终答案。"
    return {
        "question": question,
        "answer": answer_text if answer_text else explanation,
        "source": "math",
        "scoring": "keyword",
    }


# ============================================================
# 2. 数据加载器 (复用 build_calibration_data 的逻辑)
# ============================================================

def load_json(path: Path) -> list | dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return results


def load_source(raw_path: str) -> list | list[dict]:
    path = Path(raw_path)
    if not path.exists():
        path = CUSTOM_DATA_DIR / raw_path
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        elif first_char == "{":
            lines = [f.readline().strip() for _ in range(3)]
            if lines[0].startswith("{") and (len(lines) > 1 and lines[1].startswith("{") or len(lines) == 1):
                f.seek(0)
                return load_jsonl(path)
            else:
                f.seek(0)
                return [json.load(f)]
        else:
            return []


def load_samples_from_sft_dir(dir_path: Path) -> list[dict]:
    samples = []
    for f in sorted(dir_path.rglob("*.json")):
        try:
            data = json.loads(f.read_text("utf-8"))
            if isinstance(data, list):
                for item in data:
                    samples.append(item)
            elif isinstance(data, dict):
                samples.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return samples


def load_samples_from_codegen_dir(dir_path: Path, max_files: int = 3) -> list[dict]:
    samples = []
    jsonl_files = sorted(dir_path.glob("code_combine_testcases-*.json"))
    for f in jsonl_files[:max_files]:
        try:
            samples.extend(load_jsonl(f))
        except Exception:
            continue
    return samples


def load_samples_from_math_dir(dir_path: Path, max_files: int = 8) -> list[dict]:
    samples = []
    json_files = sorted(dir_path.glob("*_question_split_result.json"))
    for f in json_files[:max_files]:
        try:
            data = load_json(f)
            if isinstance(data, list):
                samples.extend(data)
            elif isinstance(data, dict):
                samples.append(data)
        except Exception:
            continue
    return samples


# ============================================================
# 3. 数据源注册
# ============================================================

DATA_SOURCES = [
    {
        "name": "comm_qa_seed",
        "paths": ["comm_qa/seed_comm_qa.json"],
        "load": lambda p: load_source(p),
        "extract": lambda obj: extract_alpaca_qa(obj),
        "weight": 0.10,
    },
    {
        "name": "comm_qa_selfinst1",
        "paths": ["comm_qa/selfinst_comm_qa_20240816.json"],
        "load": lambda p: load_source(p),
        "extract": lambda obj: extract_alpaca_qa(obj),
        "weight": 0.10,
    },
    {
        "name": "comm_qa_selfinst2",
        "paths": ["comm_qa/selfinst_comm_qa_20240818.json"],
        "load": lambda p: load_source(p),
        "extract": lambda obj: extract_alpaca_qa(obj),
        "weight": 0.15,
    },
    {
        "name": "telecom_exam",
        "paths": ["TeleQnA-exam/test_exam.json"],
        "load": lambda p: load_source(p),
        "extract": lambda obj: extract_alpaca_qa(obj),
        "weight": 0.20,
    },
    {
        "name": "spec_exam",
        "paths": ["TSpec-LLM-Q-small-exam/Sampled_3GPP_TR_Questions_exam.json"],
        "load": lambda p: load_source(p),
        "extract": lambda obj: extract_alpaca_qa(obj),
        "weight": 0.05,
    },
    {
        "name": "agent_general",
        "paths": ["agentgen/rl_data/general_tasks.json"],
        "load": lambda p: load_source(p),
        "extract_tasks": True,
        "extract": lambda obj: extract_tasks_qa(obj),
        "weight": 0.05,
    },
    {
        "name": "agent_iridium",
        "paths": ["agentgen/rl_data/iridium_tasks.json"],
        "load": lambda p: load_source(p),
        "extract_tasks": True,
        "extract": lambda obj: extract_tasks_qa(obj),
        "weight": 0.05,
    },
    {
        "name": "agent_sft",
        "paths": ["agentgen/sft_data"],
        "is_sft_dir": True,
        "load": lambda p: load_samples_from_sft_dir(p),
        "extract": lambda obj: extract_messages_qa(obj),
        "weight": 0.10,
    },
    {
        "name": "codegen",
        "paths": ["codegen/org"],
        "is_codegen_dir": True,
        "load": lambda p: load_samples_from_codegen_dir(p),
        "extract": lambda obj: extract_codegen_qa(obj),
        "weight": 0.05,
    },
    {
        "name": "math",
        "paths": ["math/train"],
        "is_math_dir": True,
        "load": lambda p: load_samples_from_math_dir(p),
        "extract": lambda obj: extract_math_qa(obj),
        "weight": 0.15,
    },
]


def main():
    parser = argparse.ArgumentParser(
        description="构建领域精度评测 Benchmark 数据集 (question-answer JSONL)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(CUSTOM_DATA_DIR.parent / "evaluation" / "accuracy_benchmark.jsonl"),
        help="输出 JSONL 路径 (默认 data/evaluation/accuracy_benchmark.jsonl)"
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=200,
        help="总样本数 (默认 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=44,
        help="随机种子 (默认 44, 与校准/评估数据集不重叠)"
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="列出可用数据源但不构建"
    )
    # 按类型分配比例（指定后只包含对应类型的数据源）
    parser.add_argument(
        "--qa-ratio", type=float, default=None,
        help="QA (alpaca) 类型占比，如 0.6 表示 60% (需同时指定 --math-ratio 或 --code-ratio)"
    )
    parser.add_argument(
        "--math-ratio", type=float, default=None,
        help="Math 类型占比，如 0.2 表示 20%"
    )
    parser.add_argument(
        "--code-ratio", type=float, default=None,
        help="Code (codegen) 类型占比，如 0.2 表示 20%"
    )
    args = parser.parse_args()
    
    random.seed(args.seed)
    
    if args.list_sources:
        print("可用数据源:")
        for src in DATA_SOURCES:
            paths = ", ".join(str(p) for p in src["paths"])
            print(f"  {src['name']:25s} weight={src['weight']:.2f}  scoring={src.get('scoring', 'keyword')}  {paths}")
        return
    
    # ---- 按类型比例过滤数据源 ----
    # 源类型 → "extract" 函数对应的输出 source 值
    TYPE_MAP = {
        "alpaca": ["comm_qa_seed", "comm_qa_selfinst1", "comm_qa_selfinst2",
            "telecom_exam", "spec_exam"],
        "math": ["math"],
        "codegen": ["codegen"],
    }
    RATIO_ARGS = {
        "alpaca": args.qa_ratio,
        "math": args.math_ratio,
        "codegen": args.code_ratio,
    }

    has_ratio = any(v is not None for v in RATIO_ARGS.values())
    if has_ratio:
        # 验证比例之和 ≈ 1.0
        total_ratio = sum(v for v in RATIO_ARGS.values() if v is not None)
        if abs(total_ratio - 1.0) > 0.01:
            print(f"[错误] 比例之和应为 1.0，当前为 {total_ratio:.2f}")
            sys.exit(1)

        # 确定要保留的源名称集合
        keep_source_names = set()
        for src_type, ratio in RATIO_ARGS.items():
            if ratio is not None and ratio > 0:
                keep_source_names.update(TYPE_MAP[src_type])

        # 过滤 DATA_SOURCES，只保留匹配的源
        # 同时调整 weight 为比例内相对权重
        filtered_sources = []
        for src in DATA_SOURCES:
            if src["name"] in keep_source_names:
                # 确定此源属于哪个类型
                for src_type, names in TYPE_MAP.items():
                    if src["name"] in names:
                        type_ratio = RATIO_ARGS[src_type]
                        # 此类型下所有源的总原始权重
                        type_total_weight = sum(
                            s["weight"] for s in DATA_SOURCES if s["name"] in names
                        )
                        if type_total_weight > 0:
                            # 按原始相对权重重新分配
                            new_weight = type_ratio * src["weight"] / type_total_weight
                            src = dict(src)
                            src["weight"] = new_weight
                        filtered_sources.append(src)
                        break

        if not filtered_sources:
            print("[错误] 过滤后没有数据源可用")
            sys.exit(1)

        print(f"按类型比例过滤数据源:")
        for src_type, ratio in RATIO_ARGS.items():
            if ratio is not None:
                count = sum(1 for s in filtered_sources
                    if s["name"] in TYPE_MAP[src_type])
                print(f"  {src_type:10s}: ratio={ratio:.0%}  sources={count}")
        print()

        # 替换 DATA_SOURCES
        sources = filtered_sources
    else:
        sources = DATA_SOURCES

    # ---- 阶段1: 加载所有数据源 ----
    print("=" * 60)
    print("加载数据源...")
    print("=" * 60)

    all_candidates = []
    source_weights = []

    for src in sources:
        name = src["name"]
        weight = src["weight"]
        if weight <= 0:
            continue

        raw_items = []

        if src.get("is_sft_dir"):
            for p in src["paths"]:
                dir_path = CUSTOM_DATA_DIR / p
                if dir_path.is_dir():
                    raw_items.extend(load_samples_from_sft_dir(dir_path))
        elif src.get("is_codegen_dir"):
            for p in src["paths"]:
                dir_path = CUSTOM_DATA_DIR / p
                if dir_path.is_dir():
                    raw_items.extend(load_samples_from_codegen_dir(dir_path))
        elif src.get("is_math_dir"):
            for p in src["paths"]:
                dir_path = CUSTOM_DATA_DIR / p
                if dir_path.is_dir():
                    raw_items.extend(load_samples_from_math_dir(dir_path))
        else:
            for p in src["paths"]:
                raw_path = CUSTOM_DATA_DIR / p
                if raw_path.exists():
                    raw_items.extend(load_source(str(raw_path)))

        extract_fn = src["extract"]
        is_tasks = src.get("extract_tasks", False)

        extracted = 0
        for item in raw_items:
            if is_tasks:
                batch = extract_fn(item)
                if batch:
                    for s in batch:
                        if s:
                            all_candidates.append((s, name))
                            extracted += 1
            else:
                result = extract_fn(item)
                if result:
                    all_candidates.append((result, name))
                    extracted += 1

        source_weights.append((name, weight, extracted))
        print(f"  {name:25s} weight={weight:.2f}  loaded={extracted}")

    total_raw = len(all_candidates)
    print(f"\n总计: {total_raw} 条候选 QA 对")

    if total_raw == 0:
        print("[错误] 没有加载到任何数据, 检查 data/custom_data/ 目录")
        sys.exit(1)

    # ---- 阶段2: 按比例分配 ----
    total_weight = sum(w for _, w, _ in source_weights)
    allocations = {}
    for name, weight, count in source_weights:
        if count == 0:
            continue
        raw_quota = int(args.num_samples * weight / total_weight)
        allocations[name] = min(raw_quota, count)

    # 补齐不足
    allocated_total = sum(allocations.values())
    if allocated_total < args.num_samples:
        remaining = args.num_samples - allocated_total
        candidates_remaining = [(name, count - allocations[name], weight / total_weight)
            for name, weight, count in source_weights
            if count > allocations.get(name, 0)]
        if candidates_remaining:
            total_rem_weight = sum(w for _, _, w in candidates_remaining)
            for name, avail, w in candidates_remaining:
                extra = int(remaining * w / total_rem_weight) if total_rem_weight > 0 else 0
                allocations[name] = allocations.get(name, 0) + min(extra, avail)

    print(f"\n配比分配 ({args.num_samples} 条):")
    for name, quota in sorted(allocations.items()):
        if quota > 0:
            available = sum(1 for _, s in all_candidates if s == name)
            print(f"  {name:25s} quota={quota:4d}  available={available}")

    # ---- 阶段3: 随机采样 ----
    selected = []
    for name, quota in allocations.items():
        if quota <= 0:
            continue
        pool = [s for s, src_name in all_candidates if src_name == name]
        random.shuffle(pool)
        selected.extend(pool[:quota])

    random.shuffle(selected)

    # ---- 阶段4: 长度过滤（确保 question 和 answer 都合理） ----
    filtered = []
    for s in selected:
        q = s.get("question", "")
        a = s.get("answer", "")
        if len(q) < 10 or len(a) < 5:
            continue
        # 过滤过长的问答对（可能包含噪声）
        if len(q) > 2000 or len(a) > 8000:
            continue
        filtered.append(s)

    final_samples = filtered[:args.num_samples]
    print(f"\n最终 Benchmark: {len(final_samples)} 条")

    # ---- 阶段5: 输出 JSONL ----
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"输出: {args.output}")
    print(f"格式: question-answer (每行一条, 含 question/answer/source/scoring 字段)")

    # ---- 统计 ----
    print(f"\n数据来源统计:")
    source_counts = {}
    scoring_counts = {}
    for s in final_samples:
        src = s.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
        sc = s.get("scoring", "keyword")
        scoring_counts[sc] = scoring_counts.get(sc, 0) + 1

    for src, count in sorted(source_counts.items()):
        print(f"  {src:20s}: {count:4d} 条")
    print(f"\n评分策略分布:")
    for sc, count in sorted(scoring_counts.items()):
        print(f"  {sc:20s}: {count:4d} 条")

    avg_q_len = sum(len(s.get("question", "")) for s in final_samples) / len(final_samples)
    avg_a_len = sum(len(s.get("answer", "")) for s in final_samples) / len(final_samples)
    print(f"\n统计:")
    print(f"  平均问题长度: {avg_q_len:.0f} 字符")
    print(f"  平均答案长度: {avg_a_len:.0f} 字符")

if __name__ == "__main__":
    main()