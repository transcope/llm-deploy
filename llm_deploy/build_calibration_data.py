#!/usr/bin/env python3
"""
	领域校准/评估数据集构建工具
	
	从 data/custom_data/ 下的多源领域数据中按比例混合采样，
	支持两种输出模式:
	  - calibration (默认): 输出 messages 格式 JSONL, 供 quantize_model.py 的
	    calibration.custom_data 字段直接使用
	  - eval: 输出 text 格式 JSONL, 供 validate_calibration.py --val-data 使用
	
	数据来源及格式支持:
  - Alpaca 格式 (JSON 数组): instruction/input/output → messages
  - messages 格式 (JSON/JSONL): 直接使用
  - tasks 格式 (JSON): {"tasks": [...]} → question/task_description → messages
  - codegen 格式 (JSONL): question/code → messages
  - math 格式 (JSON 数组): 答案讲解 → messages

	用法:
	    # 默认: 自动发现所有数据源, 产出 256 条校准集, 按默认配比混合
	    python llm_deploy/build_calibration_data.py
	
	    # 产出 100 条评估数据集 (text 格式, 供 PPL 验证)
	    python llm_deploy/build_calibration_data.py --mode eval --num-samples 100
	
	    # 自定义总样本数
	    python llm_deploy/build_calibration_data.py --num-samples 512
	
	    # 指定配比 (总比例应=1.0)
	    python llm_deploy/build_calibration_data.py \
	        --weights "comm_qa:0.35,exam:0.35,agent:0.20,math:0.10"
	
	    # 列出可用数据源
	    python llm_deploy/build_calibration_data.py --list-sources
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path


# ============================================================
# 1. 数据源发现与提取
# ============================================================

CUSTOM_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "custom_data"


def extract_alpaca(obj: dict) -> dict | None:
    """Alpaca 格式: instruction/input/output → messages"""
    inst = (obj.get("instruction") or "").strip()
    inp = (obj.get("input") or "").strip()
    out = (obj.get("output") or "").strip()
    if not inst or not out:
        return None
    # 通信领域的 exam 数据有些 instruction 本身带 "Question:" 前缀
    user_text = inst
    if inp:
        user_text += "\n" + inp
    # 答案通常在 output 的最后一句
    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": out},
        ]
    }


def extract_messages(obj: dict) -> dict | None:
    """messages 格式: 直接取 messages 字段, 要求至少有一组 user/assistant"""
    msgs = obj.get("messages")
    if not msgs or not isinstance(msgs, list) or len(msgs) < 2:
        return None
    roles = {m.get("role") for m in msgs}
    if "user" not in roles or "assistant" not in roles:
        return None
    return obj  # 原样返回


def extract_tasks(obj: dict) -> list[dict]:
    """tasks 格式: {"tasks": [{"question": ..., "task_description": ..., ...}]}"""
    results = []
    tasks = obj.get("tasks", [])
    for t in tasks:
        q = (t.get("question") or "").strip()
        desc = (t.get("task_description") or "").strip()
        gt = (t.get("ground_truth") or "").strip()
        if not q:
            continue
        # 用 task_description 作为 assistant 回答, 如果没有则用 ground_truth
        answer = desc if desc else gt
        if not answer:
            continue
        results.append({
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": answer},
            ]
        })
    return results


def extract_codegen(obj: dict) -> dict | None:
    """codegen JSONL 格式: question + code → messages"""
    q = (obj.get("question") or "").strip()
    code = (obj.get("code") or "").strip()
    if not q or not code:
        return None
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"```python\n{code}\n```"},
        ]
    }


def extract_math(obj: dict) -> dict | None:
    """math JSON 数组格式:
    {
      "category": ["计算题"],
      "answer_post": "答案文本",
      "answer_new": ["\n\\boxed{...}"],
      "explanation": "解题过程..."
    }
    """
    # math 数据没有显式的 question, 但有 explanation 和 answer
    # 用 category + "的解题过程" 作为 user 提示
    cat = (obj.get("category") or ["未知"])[0] if obj.get("category") else "未知"
    explanation = (obj.get("explanation") or "").strip()
    answer_new = obj.get("answer_new") or []
    answer_post = (obj.get("answer_post") or "").strip()

    if not explanation:
        return None

    # 从 answer_new/answer_post 取答案
    answer_text = ""
    for a in answer_new:
        if isinstance(a, str) and a.strip():
            answer_text += a.strip() + "\n"
    if not answer_text and answer_post:
        answer_text = answer_post

    user_text = f"请讲解一下{cat}类型的解题思路和步骤。"
    assistant_text = explanation
    if answer_text:
        assistant_text += f"\n\n答案: {answer_text}"

    return {
        "messages": [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
    }


# ============================================================
# 2. 数据加载器: 自动按路径/格式加载
# ============================================================

def load_json(path: Path) -> list | dict:
    """加载 JSON(数组或对象), 兼容超大文件"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict]:
    """加载 JSONL (每行一条 JSON)"""
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
    """加载一个数据源文件, 自动识别 JSON 数组 / JSONL / JSON 对象"""
    path = Path(raw_path)
    if not path.exists():
        # 尝试相对于 CUSTOM_DATA_DIR
        path = CUSTOM_DATA_DIR / raw_path
    if not path.exists():
        print(f"  [跳过] 文件不存在: {raw_path}")
        return []

    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)  # JSON array
        elif first_char == "{":
            # JSONL: 读前几行判断
            lines = [f.readline().strip() for _ in range(3)]
            if lines[0].startswith("{") and (len(lines) > 1 and lines[1].startswith("{") or len(lines) == 1):
                # 可能是 JSONL
                f.seek(0)
                return load_jsonl(path)
            else:
                # 单个 JSON 对象
                f.seek(0)
                return [json.load(f)]
        else:
            print(f"  [跳过] 未知格式: {raw_path}")
            return []


# ============================================================
# 3. 数据源注册: (路径模式, 提取函数, 权重)
# ============================================================

DATA_SOURCES = [
    # ---- 通信知识问答 (Alpaca 格式, 直接相关) ----
    {
        "name": "comm_qa_seed",
        "paths": ["comm_qa/seed_comm_qa.json"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.10,
    },
    {
        "name": "comm_qa_selfinst1",
        "paths": ["comm_qa/selfinst_comm_qa_20240816.json"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.10,
    },
    {
        "name": "comm_qa_selfinst2",
        "paths": ["comm_qa/selfinst_comm_qa_20240818.json"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.15,
    },
    # ---- 通信考试题 (Alpaca 格式, 3GPP 标准) ----
    {
        "name": "telecom_exam",
        "paths": ["TeleQnA-exam/test_exam.json"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.20,
    },
    {
        "name": "telecom_exam_jsonl",
        "paths": ["TeleQnA-exam/test_exam.jsonl"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.00,  # 和上面重复, 权重设为 0 跳过
    },
    {
        "name": "spec_exam",
        "paths": ["TSpec-LLM-Q-small-exam/Sampled_3GPP_TR_Questions_exam.json"],
        "extract": lambda obj: extract_alpaca(obj),
        "weight": 0.05,
    },
    # ---- Agent 任务规划 (tasks 格式, 信号分析领域) ----
    {
        "name": "agent_general",
        "paths": ["agentgen/rl_data/general_tasks.json"],
        "extract_tasks": True,  # 特殊: tasks 格式, 返回 list
        "extract": lambda obj: extract_tasks(obj),
        "weight": 0.05,
    },
    {
        "name": "agent_iridium",
        "paths": ["agentgen/rl_data/iridium_tasks.json"],
        "extract_tasks": True,
        "extract": lambda obj: extract_tasks(obj),
        "weight": 0.05,
    },
    # ---- Agent SFT (messages 格式, 直接可用) ----
    {
        "name": "agent_sft",
        "paths": ["agentgen/sft_data"],
        "recursive": True,
        "is_sft_dir": True,  # 特殊标记: 递归读 sft_data 所有 json
        "extract": lambda obj: extract_messages(obj),
        "weight": 0.10,
    },
    # ---- 代码生成 (codegen JSONL) ----
    {
        "name": "codegen",
        "paths": ["codegen/org"],  # 目录, 选前面几个文件
        "is_codegen_dir": True,
        "extract": lambda obj: extract_codegen(obj),
        "weight": 0.05,
    },
    # ---- 信号处理/通信数学 (math JSON 数组) ----
    {
        "name": "math",
        "paths": ["math/train"],
        "is_math_dir": True,
        "extract": lambda obj: extract_math(obj),
        "weight": 0.15,
    },
]


def load_samples_from_sft_dir(dir_path: Path) -> list[dict]:
    """递归加载 sft_data 目录下所有 json 文件中的 messages"""
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
    print(f"    SFT 目录 {dir_path.name}: 加载了 {len(samples)} 个对象")
    return samples


def load_samples_from_codegen_dir(dir_path: Path, max_files: int = 3) -> list[dict]:
    """加载 codegen 目录下的 JSONL 文件"""
    samples = []
    jsonl_files = sorted(dir_path.glob("code_combine_testcases-*.json"))
    for f in jsonl_files[:max_files]:
        try:
            lines = load_jsonl(f)
            samples.extend(lines)
            print(f"    codegen {f.name}: {len(lines)} 条")
        except Exception:
            continue
    return samples


def load_samples_from_math_dir(dir_path: Path, max_files: int = 8) -> list[dict]:
    """加载 math/train 目录下的 JSON 数组文件"""
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
# 4. 主流程
# ============================================================

def token_count_estimate(text: str) -> int:
    """估算 token 数: 中文 ~1.5char/token, 英文 ~4char/token"""
    # 粗略估计: 按中英文混合 ~2 char/token
    return len(text) // 2


def main():
    parser = argparse.ArgumentParser(
        description="构建领域校准数据集 (messages 格式 JSONL)"
    )
    parser.add_argument(
        "--output", "-o",
        default=str(CUSTOM_DATA_DIR / "calibration_data.jsonl"),
        help="输出 JSONL 路径 (默认 data/custom_data/calibration_data.jsonl)"
    )
    parser.add_argument(
        "--num-samples", "-n", type=int, default=256,
        help="总样本数 (默认 256)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子 (默认: calibration=42, eval=43)"
    )
    parser.add_argument(
        "--min-tokens", type=int, default=32,
        help="最小 token 数过滤 (默认 32)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=4096,
        help="最大 token 数过滤 (默认 4096)"
    )
    parser.add_argument(
        "--long-ratio", type=float, default=0.15,
        help="长文本(>3000 tokens)占比下限 (默认 0.15)"
    )
    parser.add_argument(
        "--list-sources", action="store_true",
        help="列出可用数据源但不构建"
    )
    parser.add_argument(
        "--mode", choices=["calibration", "eval"], default="calibration",
        help="输出模式: calibration=messages格式(默认), eval=text格式(供 PPL 验证)"
    )
    args = parser.parse_args()

    if args.seed is None:
        args.seed = 43 if args.mode == "eval" else 42
    random.seed(args.seed)

    if args.list_sources:
        print("可用的数据源:")
        for src in DATA_SOURCES:
            paths = ", ".join(str(p) for p in src["paths"])
            print(f"  {src['name']:25s} weight={src['weight']:.2f}  {paths}")
        return

    # ---- 阶段1: 加载所有数据源 ----
    print("=" * 60)
    print("加载数据源...")
    print("=" * 60)

    all_candidates = []  # [(sample, source_name), ...]
    source_weights = []

    for src in DATA_SOURCES:
        name = src["name"]
        weight = src["weight"]
        if weight <= 0:
            continue

        raw_items = []

        # 根据类型加载
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
                else:
                    print(f"  [跳过] {name}: {raw_path} 不存在")

        # 提取 messages
        extract_fn = src["extract"]
        is_tasks = src.get("extract_tasks", False)

        extracted = 0
        for item in raw_items:
            if is_tasks:
                # tasks 格式返回 list
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
    print(f"\n总计: {total_raw} 条候选样本")

    if total_raw == 0:
        print("[错误] 没有加载到任何数据, 检查 data/custom_data/ 目录")
        sys.exit(1)

    # ---- 阶段2: 按比例分配样本数 ----
    # 计算每种源的配额
    total_weight = sum(w for _, w, _ in source_weights)
    allocations = {}
    for name, weight, count in source_weights:
        if count == 0:
            continue
        raw_quota = int(args.num_samples * weight / total_weight)
        allocations[name] = min(raw_quota, count)

    # 确保总和 <= num_samples
    allocated_total = sum(allocations.values())
    if allocated_total < args.num_samples:
        # 有多余配额, 按权重分配给还有剩余样本的源
        remaining = args.num_samples - allocated_total
        candidates_remaining = [(name, count - allocations[name], weight / total_weight)
                                for name, weight, count in source_weights
                                if count > allocations.get(name, 0)]
        if candidates_remaining:
            total_rem_weight = sum(w for _, _, w in candidates_remaining)
            for name, avail, w in candidates_remaining:
                extra = int(remaining * w / total_rem_weight) if total_rem_weight > 0 else 0
                allocations[name] = allocations.get(name, 0) + min(extra, avail)

    # 打平补齐
    allocated_total = sum(allocations.values())
    if allocated_total < args.num_samples:
        # 随便找个还有余量的源补齐
        for name in allocations:
            current_count = sum(1 for _, s in all_candidates if s == name)
            if allocations[name] < current_count:
                allocations[name] += args.num_samples - allocated_total
                break

    print(f"\n配比分配 ({args.num_samples} 条):")
    for name, quota in sorted(allocations.items()):
        if quota > 0:
            available = sum(1 for _, s in all_candidates if s == name)
            print(f"  {name:25s} quota={quota:4d}  available={available}")

    # ---- 阶段3: 从各源随机采样 ----
    selected = []
    for name, quota in allocations.items():
        if quota <= 0:
            continue
        pool = [s for s, src_name in all_candidates if src_name == name]
        random.shuffle(pool)
        selected.extend(pool[:quota])

    random.shuffle(selected)

    # ---- 阶段4: 长度过滤 ----
    print(f"\n长度过滤 (min={args.min_tokens}, max={args.max_tokens}):")
    filtered = []
    for s in selected:
        msgs = s.get("messages", [])
        full_text = " ".join(m.get("content", "") for m in msgs)
        tok_est = token_count_estimate(full_text)
        if args.min_tokens <= tok_est <= args.max_tokens:
            filtered.append(s)

    print(f"  过滤前: {len(selected)} 过滤后: {len(filtered)}")

    # ---- 阶段5: 确保长文本占比 ----
    long_samples = [s for s in filtered if
                    token_count_estimate(
                        " ".join(m.get("content", "") for m in s.get("messages", []))
                    ) > 3000]
    current_long_ratio = len(long_samples) / len(filtered) if filtered else 0
    print(f"  长文本(>3000t): {len(long_samples)}/{len(filtered)} = {current_long_ratio:.1%}")

    if current_long_ratio < args.long_ratio and len(filtered) > 0:
        # 从丢弃的样本里捞长文本
        discarded = [s for s in selected if s not in filtered]
        long_discarded = [s for s in discarded if
                          token_count_estimate(
                              " ".join(m.get("content", "") for m in s.get("messages", []))
                          ) > 3000]
        needed = int(args.long_ratio * len(filtered)) - len(long_samples)
        if long_discarded and needed > 0:
            print(f"  长文本不足, 补充 {min(needed, len(long_discarded))} 条长文本")
            filtered.extend(long_discarded[:min(needed, len(long_discarded))])
            random.shuffle(filtered)

    # 最后的截断: 确保不超过 num_samples
    final_samples = filtered[:args.num_samples]
    print(f"\n最终校准集: {len(final_samples)} 条")

    # ---- 阶段6: 输出 ----
    is_eval = args.mode == "eval"

    # 自动调整输出文件名
    if args.output == str(CUSTOM_DATA_DIR / "calibration_data.jsonl") and is_eval:
        output_path = str(CUSTOM_DATA_DIR.parent / "evaluation" / "eval_data.jsonl")
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_samples:
            if is_eval:
                # eval 模式: 将对话拼接为纯文本, 输出 {"text": "..."} 格式
                msgs = sample.get("messages", [])
                # 拼接完整对话, 保留角色标记以便 PPL 评估看到完整上下文
                parts = []
                for m in msgs:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    parts.append(f"{role}: {content}")
                full_text = "\n".join(parts)
                f.write(json.dumps({"text": full_text}, ensure_ascii=False) + "\n")
            else:
                # calibration 模式: 原样输出 messages 格式
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"输出: {output_path}")
    if is_eval:
        print(f"格式: text (每行一条, 含 text 字段)")
        print(f"      可用于 validate_calibration.py --val-data")
    else:
        print(f"格式: messages (每行一条, 含 messages 字段)")
        print(f"      可直接用于 calibration.custom_data")

    # ---- 阶段7: 打印统计 ----
    print(f"\n数据来源统计:")
    source_stats = {}
    for s in final_samples:
        # 追溯来源: 用第一个 user message 的前50字匹配? 太复杂, 直接按来源统计
        pass

    # 估算 token 分布
    token_counts = []
    for s in final_samples:
        full_text = " ".join(m.get("content", "") for m in s.get("messages", []))
        token_counts.append(token_count_estimate(full_text))

    if token_counts:
        print(f"  Token 分布 (估计): min={min(token_counts)}, "
              f"max={max(token_counts)}, "
              f"avg={sum(token_counts)//len(token_counts)}")

    # 按原始权重分组统计
    print(f"\n建议的 yaml calibration 段落:")
    print(f"```yaml")
    print(f"calibration:")
    print(f"  custom_data: \"{args.output}\"")
    print(f"  num_samples: {len(final_samples)}")
    print(f"  hf_offline: true")
    print(f"```")


if __name__ == "__main__":
    main()
