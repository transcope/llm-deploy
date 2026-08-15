# 评估协议 —— 基于领域数据集的量化质量验证

> ⚠️ **本文档为当前唯一的精度评测标准**。早期基于 lm-eval 的标准 Benchmark 精度评测
> （`benchmark_eval.py --tasks`，GSM8K/HellaSwag 等）仅用于最初可行性验证，**已弃用**。
> 精度评测统一使用本文档定义的领域精度评测（`benchmark_domain.py`）。

> 本文档定义了一套完整的领域评估方案，包含两个互补维度：
>
> 1. **PPL 快速验证**（perplexity）—— 分钟级，衡量模型对领域文本的"困惑度"，快速发现校准数据质量问题
> 2. **精度评测**（task accuracy）—— 小时级，衡量模型在领域 QA 任务上的正确率，反映实际业务能力
>
> 适用范围：所有需要校准的量化方案（GPTQ / AWQ / FP8 / W8A8）的量化后验证。

---

## 目录

- [1. 评估流程概览](#1-评估流程概览)
- [2. 环境准备与安装](#2-环境准备与安装)
- [3. PPL 评估数据集](#3-ppl-评估数据集)
- [4. 精度评测 Benchmark 数据集](#4-精度评测-benchmark-数据集)
- [5. 基线测量 (PPL)](#5-基线测量-ppl)
- [6. 量化后验证 (PPL)](#6-量化后验证-ppl)
- [7. 领域精度评测](#7-领域精度评测)
- [8. 结果解读与判定标准](#8-结果解读与判定标准)
- [9. 模型特定预期值](#9-模型特定预期值)
- [10. 完整命令速查](#10-完整命令速查)
- [11. 故障排查](#11-故障排查)

---

## 1. 评估流程概览

```
                      ┌──────────────────────────┐
                      │   data/custom_data/ 领域数据 │
                      └──────────┬───────────────┘
                                 │
                  ┌──────────────┼──────────────────────────┐
                  ▼              ▼                          ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────────┐
        │  build_calib-    │ │ build_accuracy_ │ │ build_calibration_   │
        │  ration.py       │ │ benchmark.py    │ │ data.py --mode eval  │
        │  --mode calibration               │ │                      │
        └────────┬────────┘ └────────┬────────┘ └──────────┬───────────┘
                 ▼                   ▼                     ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
        │ calibration_    │ │ accuracy_bench- │ │ eval_data.jsonl  │
        │ data.jsonl      │ │ mark.jsonl      │ │ (PPL, 100条)      │
        │ (量化校准用)      │ │ (QA对, 200-300条)│ └────────┬─────────┘
        └────────┬────────┘ └────────┬────────┘          │
                 │                   │                   │
                 ▼                   ▼                   ▼
        ┌─────────────────┐ ┌─────────────────┐ ┌──────────────────┐
        │ quantize_       │ │ benchmark_      │ │ validate_calib-  │
        │ model.py        │ │ domain.py       │ │ ration.py        │
        │                 │ │ 精度评测          │ │ PPL 验证          │
        └─────────────────┘ └─────────────────┘ └──────────────────┘
                                      │                   │
                                      ▼                   ▼
                                ┌────────────┐     ┌────────────┐
                                │ Task acc % │     │ Δ PPL      │
                                │ (0~1)      │     │ (基线-量化)  │
                                └────────────┘     └────────────┘
```

### 两种评估维度对比

| 维度 | PPL 快速验证 | 领域精度评测 |
|:----:|:------------:|:------------:|
| 耗时（7B, V100） | ~10 分钟 | ~30 分钟 ~ 2 小时 |
| 衡量目标 | 模型"困惑度" | 任务"正确率" |
| 是否需标准答案 | ❌ 不需要 | ✅ 需要（question-answer pair） |
| 发现能力 | 校准数据质量、严重量化损失 | 实际业务能力退化 |
| 结果可读性 | 单个数字 (PPL) | 百分比准确率 + 按来源细分 |
| 定位 | 快速过滤器 | 最终验收标准 |

**推荐策略**：先跑 PPL 快速验证（确认校准数据和量化链路正常），再跑领域精度评测（确认业务能力达标）。

---

## 2. 环境准备与安装

### 2.1 Python 依赖

#### PPL 快速验证（`validate_calibration.py`）

```bash
pip install torch transformers accelerate
```

| 包 | 版本要求 | 说明 |
|:---|:--------:|:-----|
| `torch` | ≥ 2.0.0 | 建议按 CUDA 版本从 pytorch.org 安装 |
| `transformers` | ≥ 4.38.0 | HuggingFace 模型加载 |
| `accelerate` | ≥ 0.27.0 | 设备映射（非必需，但建议安装） |

#### 领域精度评测（`benchmark_domain.py`）

```bash
# transformers 后端（V100 推荐）
pip install torch transformers accelerate

# vLLM 后端（A100 可选）
pip install vllm
```

| 包 | 版本要求 | 说明 |
|:---|:--------:|:-----|
| `torch` | ≥ 2.0.0 | 基础框架 |
| `transformers` | ≥ 4.38.0 | transformers 后端（V100 推荐） |
| `vllm` | ≥ 0.4.0 | vLLM 后端（A100 可用，V100+Qwen3 不兼容） |

### 2.2 V100 环境适配要点

| 要点 | 说明 |
|:----|:------|
| **精度** | V100 不支持 bfloat16，必须使用 `float16` 或 `float32` |
| **Qwen3 + vLLM** | V100（SM 7.0）上 vLLM 加载 Qwen3 会崩溃（`LLVM ERROR: Failed to compute parent layout`）。请使用 `--backend transformers` |
| **device_map** | 单卡加载时用 `device_map=None` + `.to("cuda")`，不要用 `device_map="auto"`（会触发 accelerate 钩子引入额外开销） |
| **显存** | 8B 模型 fp16 约需 16GB，单卡 V100 32GB 足够 |

### 2.3 Qwen3 Thinking 模式说明

Qwen3 模型默认输出中文思维链（CoT），会占用全部 max_tokens 导致答案为空。

**解决方案**：生成时设置 `enable_thinking=False`：

```python
tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True,
                              enable_thinking=False)
```

使用 `benchmark_domain.py` 时通过 `--no-thinking` 标志自动处理（transformers 后端默认启用）。

### 2.4 数据集目录结构

```
data/
├── custom_data/          # 原始领域数据（只读，不动）
│   ├── comm_qa/
│   ├── TeleQnA-exam/
│   ├── TSpec-LLM-Q-small-exam/
│   ├── agentgen/
│   ├── codegen/
│   └── math/
├── evaluation/            # 评估数据集（脚本输出位置）
│   ├── accuracy_benchmark.jsonl   # 精度评测数据集
│   └── eval_data.jsonl            # PPL 评估数据集
└── calibration/           # 校准数据集（量化用）
```

### 2.5 容器环境确认

```bash
# 进入容器后确认环境
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
# 预期输出: CUDA: True, GPU: Tesla V100S-PCIE-32GB
```

---

## 3. PPL 评估数据集

### 3.1 构建命令

```bash
# 从领域数据构建 100 条评估集（text 格式，供 PPL 验证使用）
python llm_deploy/build_calibration_data.py \
    --mode eval \
    --num-samples 100 \
    --seed 43
```

### 3.2 输出格式

```jsonl
{"text": "user: 什么是 5G 核心网 SBA 架构？\nassistant: SBA（Service-Based Architecture）是 5G 核心网的新型架构..."}
{"text": "user: 请解释一下什么是量子计算。\nassistant: 量子计算是一种利用量子力学原理..."}
```

每行一个 JSON 对象，包含 `text` 字段，值为完整对话文本（`user: ...\nassistant: ...`）。

### 3.3 数据规格

| 规格 | 值 |
|------|-----|
| 样本数 | 100 条 |
| 数据源 | 10 个领域源（通信 55% / 数学 15% / Agent 20% / 代码 5%） |
| Token 范围（估计） | 67 ~ 37318 |
| 平均 Token（估计） | ~2809 |
| 输出位置 | `data/evaluation/eval_data.jsonl` |

### 3.4 与校准数据集的关系

| 维度 | 校准集 (`calibration_data.jsonl`) | 评估集 (`eval_data.jsonl`) |
|:----:|:---------------------------------:|:--------------------------:|
| 用途 | 量化阶段统计激活分布 | 量化后验证精度损失 |
| 格式 | `{"messages": [...]}` | `{"text": "..."}` |
| 消费方 | `quantize_model.py` | `validate_calibration.py --val-data` |
| 构建模式 | `--mode calibration` | `--mode eval` |
| 默认随机种子 | 42 | 43（不重叠） |

---

## 4. 精度评测 Benchmark 数据集

### 4.1 构建命令

```bash
# 从领域数据构建精度评测 Benchmark（格式，供 benchmark_domain.py 使用）
python llm_deploy/build_accuracy_benchmark.py \
    --num-samples 300 \
    --seed 44
```

参数说明：

| 参数 | 默认值 | 说明 |
|:----|:------:|:-----|
| `--num-samples / -n` | 200 | 目标样本数（最终数量受数据源可用数限制） |
| `--seed` | 44 | 随机种子（与校准/评估数据集不重叠） |
| `--output / -o` | `data/evaluation/accuracy_benchmark.jsonl` | 输出路径 |
| `--list-sources` | (flag) | 列出可用数据源但不构建 |

### 4.2 输出格式

```jsonl
{"question": "什么是 5G 核心网 SBA 架构？请详细解释。", "answer": "SBA（Service-Based Architecture）是 5G 核心网采用的一种基于服务的架构...", "source": "alpaca", "scoring": "keyword"}
{"question": "请根据以下需求生成 Python 代码：读取 CSV 文件并计算平均值", "answer": "```python\nimport pandas as pd\n...\n```", "source": "codegen", "scoring": "keyword"}
{"question": "计算 ∫(0→π) sin(x) dx", "answer": "2", "source": "math", "scoring": "keyword"}
```

每行包含：

| 字段 | 类型 | 说明 |
|:----|:----:|:-----|
| `question` | string | 问题（用于提问模型） |
| `answer` | string | 标准答案（用于评分对比） |
| `source` | string | 数据来源类型：`alpaca` / `tasks` / `messages` / `codegen` / `math` |
| `scoring` | string | 推荐评分策略：`keyword`（关键词召回） / `exact_match`（精确匹配） |

### 3.3 数据规格

| 规格 | 值 |
|------|-----|
| 样本数 | ~200-300 条（自动按比例分配后过滤） |
| 数据源 | 10 个领域源（权重分配见下表） |
| 平均问题长度 | ~200 字符 |
| 平均答案长度 | ~500 字符 |
| 评分策略 | `keyword` ~90%, `exact_match` ~10% |
| 输出位置 | `data/evaluation/accuracy_benchmark.jsonl` |

### 3.4 数据来源权重分配

| 数据源 | 权重 | 格式类型 | 评分策略 | 内容领域 |
|:------:|:----:|:--------:|:--------:|:---------|
| comm_qa_seed | 10% | alpaca | keyword | 通信 QA |
| comm_qa_selfinst1 | 10% | alpaca | keyword | 通信 QA |
| comm_qa_selfinst2 | 15% | alpaca | keyword | 通信 QA |
| telecom_exam | 20% | alpaca | keyword | 通信考试题 |
| spec_exam | 5% | alpaca | keyword | 3GPP 规范题 |
| agent_general | 5% | tasks | exact_match | Agent 通用任务 |
| agent_iridium | 5% | tasks | exact_match | Agent Iridium 任务 |
| agent_sft | 10% | messages | keyword | Agent SFT 数据 |
| codegen | 5% | codegen | keyword | 代码生成 |
| math | 15% | math | keyword | 数学解题 |

### 3.5 Benchmark 构建规则

Benchmark 从各个数据源提取 QA 对的逻辑：

| 数据格式 | 提取方式 |
|:--------:|:---------|
| **Alpaca** (`instruction/input/output`) | `instruction + "\n" + input` → question, `output` → answer |
| **Messages** (`[user, assistant, ...]`) | 第一轮 `user` → question, 第一轮 `assistant` → answer |
| **Tasks** (`question + ground_truth`) | `question` → question, `ground_truth` → answer（短答案用 exact_match） |
| **Codegen** (`question + code`) | `question` → question, `code` → answer |
| **Math** (`explanation + \boxed{...}`) | category 提示 → question, `\boxed{}` 内容 → answer |

---

## 5. 基线测量 (PPL)

### 5.1 目的

在量化之前，先测量原始模型在领域评估集上的 PPL 作为**基线值**。此值用于：
- 验证评估数据集格式正确、模型加载正常
- 为后续量化模型 PPL 提供对比基准
- 记录模型在领域数据上的"原始困惑度"

### 5.2 执行命令

```bash
python llm_deploy/validate_calibration.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized /app/local_models/Mind-SLLM-Qwen3-8B \
    --num-samples 100 \
    --val-data ./data/evaluation/eval_data.jsonl \
    --max-ppl-delta 1.0 \
    --dtype float16 \
    --output ./results/baseline_ppl.json
```

### 5.3 预期结果

由于 baseline 和 quantized 指向同一模型，PPL 应几乎一致：

```
Baseline (FP16/BF16):     ppl = 8.5000
量化模型 (auto):          ppl = 8.5008
Delta:                     +0.0008  (阈值: 1.0)
验证: ✅ 通过
```

delta < 0.01 说明环境和数据集正常。

### 5.4 结果文件

```json
{
  "baseline_ppl": 8.5,
  "quantized_ppl": 8.5008,
  "delta": 0.0008,
  "passed": true,
  "threshold": 1.0,
  "error": ""
}
```

---

## 6. 量化后验证 (PPL)

### 6.1 目的

量化完成后，对比量化模型与原始模型的 PPL 差异，判断量化是否引入了不可接受的精度损失。

### 6.2 执行命令

```bash
# 方式 A：量化时自动验证（推荐）
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --validate \
    --max-ppl-delta 5.0 \
    --val-data ./data/evaluation/eval_data.jsonl

# 方式 B：量化完成后单独验证
python llm_deploy/validate_calibration.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq \
    --num-samples 100 \
    --val-data ./data/evaluation/eval_data.jsonl \
    --dtype float16 \
    --max-ppl-delta 5.0 \
    --output ./results/quantized_validation.json
```

### 6.3 验证流程

```
加载原始模型 (FP16/BF16)
  └── 逐条计算 PPL → PPL_baseline
加载量化模型 (INT4/FP8)
  └── 逐条计算 PPL → PPL_quantized
比较 delta = PPL_quantized - PPL_baseline
  ├── delta ≤ threshold → ✅ 通过
  └── delta > threshold → ⚠️ 警告（不阻断）
```

### 6.4 自动后端映射

`quantize_model.py --validate` 自动将各种量化后端映射为 `validate_calibration.py` 识别的量化方式：

| 量化后端 | `quantization` 参数 |
|:--------:|:-------------------:|
| gptqmodel | `gptq` |
| llmcompressor (GPTQ) | `compressed-tensors` |
| llmcompressor (AWQ) | `awq` |
| llmcompressor (FP8) | `fp8` |
| llmcompressor (W8A8) | `compressed-tensors` |

---

## 7. 领域精度评测

> **定位**: 领域精度评测是量化质量的**最终验收标准**。PPL 验证通过后，应执行本评测确认业务能力达标。

### 7.1 评测原理

领域精度评测通过在 domain-specific Benchmark 数据集上比较模型输出与标准答案的一致性，量化模型在具体业务场景中的表现。

**评分策略**：

| 评分类型 | 适用场景 | 方法 |
|:--------:|:---------|:-----|
| `keyword` | 长文本答案（讲解、代码、解题过程） | 提取标准答案中的关键词（中文词组 + 英文关键词 + 数值），计算模型答案中的关键词召回率 |
| `exact_match` | 短答案（Agent 任务、确定性回答） | 数值匹配（60%权重）+ 关键词匹配（40%权重），适合有确定答案的场景 |

**评分阈值**：

| 得分范围 | 评级 | 含义 |
|:--------:|:----:|:------|
| ≥ 0.70 | ✅ 优秀 | 模型答案覆盖了大部分关键信息 |
| 0.50 ~ 0.69 | ✅ 良好 | 核心要点基本覆盖 |
| 0.35 ~ 0.49 | ⚠️ 边缘 | 部分关键信息遗漏（默认通过阈值为 0.35） |
| < 0.35 | 🔴 不及格 | 答案偏离标准答案较多 |

> 阈值 `0.35` 的设定考虑了点：关键词召回率 35% 意味着模型覆盖了超过三分之一的核心概念。对于长答案（如代码生成、技术讲解），这不是一个严格门槛，而是用于**检测严重退化**——如果量化导致模型输出质量显著下降，关键词召回率会明显低于此线。

### 7.2 运行方式

#### 方式 A：API 模式（推荐，已部署服务）

```bash
# 评测量化模型
python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8000 \
    --model Mind-SLLM-Qwen3-8B-GPTQ \
    --benchmark data/custom_data/accuracy_benchmark.jsonl \
    --output results/domain_eval_quantized.json

# 评测基线模型（需另启一个服务实例）
python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8001 \
    --model Mind-SLLM-Qwen3-8B \
    --benchmark data/custom_data/accuracy_benchmark.jsonl \
    --output results/domain_eval_baseline.json
```

#### 方式 B：本地模式（直接加载模型，需 GPU）

```bash
# 直接加载模型评测（建议在容器内执行）
python llm_deploy/benchmark_domain.py \
    --local \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --benchmark data/custom_data/accuracy_benchmark.jsonl \
    --output results/domain_baseline.json

# 评测基线 + 量化模型对比
python llm_deploy/benchmark_domain.py \
    --local \
    --model ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq \
    --benchmark data/custom_data/accuracy_benchmark.jsonl \
    --output results/domain_quantized.json
```

> **注意**: 本地模式使用 vLLM 的 `LLM.chat()` 接口，需要 GPU 显存。V100 上 8B 模型需要约 16GB 显存（FP16）或 4GB（GPTQ）。

### 7.3 参数说明

| 参数 | 默认值 | 说明 |
|:----|:------:|:-----|
| `--benchmark / -b` | `data/evaluation/accuracy_benchmark.jsonl` | Benchmark 数据集路径 |
| `--num-samples / -n` | 0（全部） | 采样数，用于快速验证 |
| `--base-url` | `http://localhost:8000` | API 模式：服务地址 |
| `--model` | `default` | API 模式：模型名称；本地模式：模型路径 |
| `--local` | (flag) | 启用本地加载模式 |
| `--tp` | 1 | 本地模式：张量并行数 |
| `--max-tokens` | 1024 | 最大生成 token 数 |
| `--temperature` | 0.0 | 生成温度（0 = 贪婪解码） |
| `--timeout` | 120 | API 超时秒数 |
| `--delay` | 0.0 | API 请求间隔秒数 |
| `--pass-threshold` | 0.35 | 单题通过阈值 |
| `--output / -o` | (stdout) | 结果输出 JSON 文件 |

### 7.4 输出说明

```json
{
  "meta": {
    "mode": "api",
    "base_url": "http://192.168.192.186:8000",
    "model": "Mind-SLLM-Qwen3-8B-GPTQ",
    "benchmark": "data/custom_data/accuracy_benchmark.jsonl",
    "num_samples": 268,
    "pass_threshold": 0.35,
    "timestamp": "2026-08-04 15:30:00"
  },
  "overall": {
    "accuracy": 0.7239,
    "avg_score": 0.5812,
    "correct": 194,
    "total": 268
  },
  "per_source": {
    "alpaca": {
      "accuracy": 0.7425,
      "avg_score": 0.5943,
      "correct": 124,
      "total": 167
    },
    "codegen": {
      "accuracy": 0.6875,
      "avg_score": 0.5210,
      "correct": 11,
      "total": 16
    },
    "math": {
      "accuracy": 0.6512,
      "avg_score": 0.5189,
      "correct": 28,
      "total": 43
    }
  },
  "results": [
    {
      "index": 0,
      "question": "...",
      "ground_truth": "...",
      "model_answer": "...",
      "score": 0.85,
      "scoring": "keyword",
      "source": "alpaca",
      "passed": true,
      "details": {
        "kw_recall": 0.85,
        "matched_keywords": ["5G", "核心网", "SBA"],
        "total_keywords": 12,
        "matched_count": 10
      }
    }
  ]
}
```

### 7.5 基线 vs 量化对比方法

要正确衡量量化带来的精度损失，需要**分别评测基线模型和量化模型**，然后对比：

```bash
# 1. 评测基线模型
python llm_deploy/benchmark_domain.py \
    --local \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --output results/domain_baseline.json

# 2. 评测量化模型（需先量化）
python llm_deploy/benchmark_domain.py \
    --local \
    --model ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --output results/domain_quantized.json

# 3. 对比结果
python -c "
import json
b = json.load(open('results/domain_baseline.json'))
q = json.load(open('results/domain_quantized.json'))
print(f'基线准确率:     {b[\"overall\"][\"accuracy\"]:.2%}')
print(f'量化准确率:     {q[\"overall\"][\"accuracy\"]:.2%}')
print(f'精度变化:       {q[\"overall\"][\"accuracy\"] - b[\"overall\"][\"accuracy\"]:+.2%}')
print()
print('按来源对比:')
for src in b['per_source']:
    b_acc = b['per_source'][src]['accuracy']
    q_acc = q['per_source'].get(src, {}).get('accuracy', 0)
    print(f'  {src:20s}: 基线 {b_acc:.2%} → 量化 {q_acc:.2%} (Δ {q_acc-b_acc:+.2%})')
"
```

---

## 8. 结果解读与判定标准

### 8.1 PPL delta 分级

| Δ PPL | 评级 | 建议 |
|:-----:|:----:|:-----|
| < 1.0 | ✅ 优秀 | 量化几乎无损，可直接部署 |
| 1.0 ~ 3.0 | ✅ 良好 | 正常范围，推荐部署 |
| 3.0 ~ 5.0 | ⚠️ 可接受 | 建议检查校准数据质量，可部署但需关注业务精度 |
| > 5.0 | 🔴 超阈值 | 建议检查校准数据、num_samples 或量化参数 |

### 8.2 领域精度评分标准

| 量化后准确率对比基线 | 评级 | 建议 |
|:-------------------:|:----:|:-----|
| 下降 < 5% | ✅ 优秀 | 量化对业务能力几乎无影响 |
| 下降 5% ~ 10% | ✅ 可接受 | 正常量化损失，推荐部署 |
| 下降 10% ~ 20% | ⚠️ 需关注 | 建议检查校准数据质量或考虑高精度方案（W8A8） |
| 下降 > 20% | 🔴 需整改 | 量化方案需重新评估 |

### 8.3 各量化方案的典型 PPL 偏移

| 量化方案 | 典型 Δ PPL | 说明 |
|:--------:|:----------:|:-----|
| W8A8 (SmoothQuant) | +0.1 ~ +0.3 | 精度损失最小 |
| AWQ W4A16 | +0.3 ~ +1.0 | 推荐方案，精度权衡好 |
| GPTQ W4A16 (desc_act=true) | +0.5 ~ +2.0 | V100 推荐方案 |
| GPTQ W4A16 (desc_act=false) | +1.0 ~ +4.0 | 兼容性优先但精度略低 |
| FP8 (H100+) | +0.1 ~ +0.5 | 几乎无损 |

### 8.4 各量化方案的典型领域精度影响

| 量化方案 | 典型准确率下降 | 适用场景 |
|:--------:|:--------------:|:---------|
| W8A8 | < 3% | 精度敏感场景 |
| AWQ W4A16 | 2% ~ 5% | 通用生产（A100+） |
| GPTQ W4A16 | 3% ~ 8% | V100 生产推荐 |
| BitsAndBytes NF4 | 2% ~ 6% | 快速验证/显存受限 |

### 8.5 重要限制

- **PPL ≠ 任务精度**：PPL 低不保证任务准确率高，反之亦然。两种评估互补而非替代
- **领域偏移**：评估数据集领域与模型实际使用领域差异越大，评估参考价值越低
- **关键词评分的局限性**：`keyword` 评分只衡量关键词召回率，不衡量语义正确性。对于需要精确推理的题目（如数学计算），可能误判
- **参考值非绝对**：不同模型的准确率绝对值不可跨模型比较，只看 delta

---

## 9. 模型特定预期值

### 9.1 Mind-SLLM-Qwen3-8B

#### PPL 预期

| 指标 | 预期范围 | 说明 |
|:----|:--------:|:-----|
| 原始模型 PPL（领域评估集） | 8.0 ~ 14.0 | 具体取决于评估集的领域匹配度 |
| GPTQ W4A16 Δ PPL | +0.5 ~ +2.5 | desc_act=true, group_size=128 |
| 单趟 PPL 耗时（V100） | ~10 min | 100 条评估文本 |

#### 领域精度预期

| 指标 | 预期范围 | 说明 |
|:----|:--------:|:-----|
| 原始模型准确率（关键词召回） | 60% ~ 85% | 取决于数据源难度 |
| GPTQ W4A16 准确率下降 | -3% ~ -8% | vs 基线模型 |
| 单次评测耗时（V100, API） | ~30 min | 300 条 benchmark |
| 单次评测耗时（V100, 本地） | ~45 min | 含模型加载时间 |

### 9.2 通用预期参考

#### PPL

| 模型规模 | 通用 PPL 范围 | 说明 |
|:--------:|:-------------:|:-----|
| 7B / 8B | 8 ~ 15 | 常见范围 |
| 14B | 7 ~ 12 | 更大模型通常 PPL 更低 |
| 32B | 6 ~ 10 | |
| 72B | 5 ~ 9 | |

#### 领域精度

| 模型规模 | 通用准确率范围 | 说明 |
|:--------:|:--------------:|:-----|
| 7B / 8B | 55% ~ 80% | 关键词召回率 |
| 14B | 60% ~ 85% | |
| 32B | 65% ~ 90% | |

---

## 10. 完整命令速查

### 10.1 环境准备

```bash
# 进入容器
docker exec -it zetta_ld bash
source /app/venv/bin/activate
cd /volume/workspace/llm-deploy
```

### 10.2 构建数据集

```bash
# PPL 评估集 (100 条)
python llm_deploy/build_calibration_data.py --mode eval --num-samples 100 --seed 43

# 精度评测 Benchmark (~300 条)
python llm_deploy/build_accuracy_benchmark.py --num-samples 300 --seed 44

# 查看 Benchmark 数据源
python llm_deploy/build_accuracy_benchmark.py --list-sources
```

### 10.3 PPL 基线测量

```bash
python llm_deploy/validate_calibration.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized /app/local_models/Mind-SLLM-Qwen3-8B \
    --num-samples 100 \
    --val-data ./data/evaluation/eval_data.jsonl \
    --max-ppl-delta 1.0 \
    --dtype float16 \
    --output ./results/baseline_ppl.json
```

### 10.4 量化 + PPL 验证

```bash
# 方式 A：量化时自动验证
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --validate \
    --max-ppl-delta 5.0

# 方式 B：量化后单独验证
python llm_deploy/validate_calibration.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq \
    --num-samples 100 \
    --val-data ./data/evaluation/eval_data.jsonl \
    --dtype float16 \
    --max-ppl-delta 5.0 \
    --output ./results/quantized_validation.json
```

### 10.5 领域精度评测

```bash
# ===== API 模式（服务已部署） =====

# 评测量化模型
python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8000 \
    --model Mind-SLLM-Qwen3-8B-GPTQ \
    --output results/domain_quantized.json

# 评测基线模型（需另启服务）
python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8001 \
    --model Mind-SLLM-Qwen3-8B \
    --output results/domain_baseline.json

# 快速验证（只测 20 条）
python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8000 \
    --model Mind-SLLM-Qwen3-8B-GPTQ \
    --num-samples 20 \
    --output results/domain_smoke.json

# ===== 本地模式（直接加载模型） =====

# 基线模型评测
python llm_deploy/benchmark_domain.py \
    --local \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --output results/domain_baseline.json

# 量化模型评测
python llm_deploy/benchmark_domain.py \
    --local \
    --model ./models/Mind-SLLM-Qwen3-8B-GPTQ \
    --output results/domain_quantized.json

# 使用多卡张量并行加速
python llm_deploy/benchmark_domain.py \
    --local \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --tp 8 \
    --output results/domain_baseline_tp8.json
```

### 10.6 对比分析

```bash
# PPL 对比
python -c "
import json
b = json.load(open('./results/baseline_ppl.json'))
q = json.load(open('./results/quantized_validation.json'))
print(f'Baseline PPL:        {b[\"baseline_ppl\"]:.4f}')
print(f'Quantized PPL:       {q[\"quantized_ppl\"]:.4f}')
print(f'△PPL (quantized):    {q[\"delta\"]:+.4f}')
print(f'Baseline → Quantized △: {q[\"quantized_ppl\"] - b[\"baseline_ppl\"]:+.4f}')
print(f'Passed:              {q[\"passed\"]}')
"

# 领域精度对比
python -c "
import json
b = json.load(open('results/domain_baseline.json'))
q = json.load(open('results/domain_quantized.json'))
print(f'基线准确率:     {b[\"overall\"][\"accuracy\"]:.2%}')
print(f'量化准确率:     {q[\"overall\"][\"accuracy\"]:.2%}')
print(f'精度变化:       {q[\"overall\"][\"accuracy\"] - b[\"overall\"][\"accuracy\"]:+.2%}')
print()
print('按来源对比:')
for src in sorted(set(list(b['per_source'].keys()) + list(q['per_source'].keys()))):
    b_acc = b['per_source'].get(src, {}).get('accuracy', 0)
    q_acc = q['per_source'].get(src, {}).get('accuracy', 0)
    print(f'  {src:20s}: 基线 {b_acc:.2%} → 量化 {q_acc:.2%} (Δ {q_acc-b_acc:+.2%})')
"

# 联合报告
python -c "
import json
b_ppl = json.load(open('./results/baseline_ppl.json'))
q_ppl = json.load(open('./results/quantized_validation.json'))
b_dom = json.load(open('results/domain_baseline.json'))
q_dom = json.load(open('results/domain_quantized.json'))

print('========== 量化质量综合报告 ==========')
print(f'模型: Mind-SLLM-Qwen3-8B → GPTQ W4A16')
print()
print('--- PPL 快速验证 ---')
print(f'  Baseline PPL:  {b_ppl[\"baseline_ppl\"]:.4f}')
print(f'  Quantized PPL: {q_ppl[\"quantized_ppl\"]:.4f}')
print(f'  Δ PPL:         {q_ppl[\"delta\"]:+.4f}')
print(f'  状态:          {\"✅ 通过\" if q_ppl[\"passed\"] else \"🔴 超阈值\"}')
print()
print('--- 领域精度评测 ---')
print(f'  Baseline Acc:  {b_dom[\"overall\"][\"accuracy\"]:.2%}')
print(f'  Quantized Acc: {q_dom[\"overall\"][\"accuracy\"]:.2%}')
print(f'  Δ Acc:         {q_dom[\"overall\"][\"accuracy\"] - b_dom[\"overall\"][\"accuracy\"]:+.2%}')
print(f'  状态:          {\"✅ 通过\" if q_dom[\"overall\"][\"accuracy\"] >= b_dom[\"overall\"][\"accuracy\"] - 0.10 else \"⚠️ 需关注\"}')
"
```

---

## 11. 故障排查

### 11.1 常见错误

| 错误 | 原因 | 解决 |
|:----|:-----|:------|
| `ModuleNotFoundError: lm_eval` | 领域评测不依赖 lm-eval（仅 requests/vllm/transformers）；若出现说明误用了已弃用的 benchmark_eval.py 精度评测 | 改用 `benchmark_domain.py` |
| `CUDA out of memory` | 显存不足 | 加 `--dtype float16`；或减少 `--num-samples` |
| PPL = NaN 或 Inf | 评估文本中有模型不支持的特殊 token | 检查 `eval_data.jsonl` 内容 |
| `trust_remote_code` 错误 | 模型架构需要远程代码 | 已在代码中内置 `trust_remote_code=true` |
| `bfloat16 not supported` | V100 不支持 bfloat16 | 指定 `--dtype float16` |
| PPL 极高（> 100） | 评估数据与模型训练分布严重不匹配 | 检查领域偏移，或使用通用评估文本 |
| API 连接超时 | 服务未启动或地址错误 | 检查 `--base-url`，确认 `curl <url>/v1/models` 可访问 |
| Benchmark 为空 | 自定义数据未加载 | 确认 `data/custom_data/` 下有源数据文件 |
| 模型回答为空 | API 返回格式错误或 Qwen3 thinking 模式未关闭 | 检查服务日志；本地模式加 `--no-thinking` |
| `LLVM ERROR: Failed to compute parent layout` | V100 + Qwen3 + vLLM 不兼容 | 改用 `--backend transformers`（V100 上 vLLM 不支持 Qwen3） |
| 模型回答全是思维链（中文推理） | Qwen3 thinking 模式未抑制 | 本地模式使用 `--backend transformers`（默认抑制）；API 模式需服务端设置 `enable_thinking=False` |
| `device_map="auto"` 导致生成极慢 | accelerate 钩子引入额外开销 | 改用 `device_map=None` + `.to("cuda")` 单卡加载 |
| 容器内 SSH 管道中断（中文乱码） | Windows 端文件传输编码问题 | 使用 `cat 文件路径 | ssh ... docker exec -i ... cat > 目标路径` 而非 `type` 或重定向 |

### 11.2 验证链路是否正常

```bash
# PPL 快速验证（5 条）
head -5 ./data/evaluation/eval_data.jsonl > /tmp/quick_test.jsonl

python llm_deploy/validate_calibration.py \
    --baseline /app/local_models/Mind-SLLM-Qwen3-8B \
    --quantized /app/local_models/Mind-SLLM-Qwen3-8B \
    --num-samples 5 \
    --val-data /tmp/quick_test.jsonl \
    --dtype float16 \
    --max-ppl-delta 1.0

# 领域精度快速验证（5 条，API 模式）
head -5 ./data/evaluation/accuracy_benchmark.jsonl > /tmp/quick_acc.jsonl

python llm_deploy/benchmark_domain.py \
    --base-url http://192.168.192.186:8000 \
    --model Mind-SLLM-Qwen3-8B-GPTQ \
    --benchmark /tmp/quick_acc.jsonl \
    --num-samples 5
```

### 11.3 Benchmark 数据问题排查

```bash
# 检查 Benchmark 文件格式
python -c "
import json
with open('data/evaluation/accuracy_benchmark.jsonl') as f:
    lines = [json.loads(l) for l in f if l.strip()]
print(f'总条数: {len(lines)}')
print(f'字段:   {list(lines[0].keys())}')
print(f'来源:   {set(l[\"source\"] for l in lines)}')
print(f'评分:   {set(l[\"scoring\"] for l in lines)}')
print(f'第一行: {json.dumps(lines[0], ensure_ascii=False)[:200]}')
"

# 重新构建（如数据源有更新）
python llm_deploy/build_accuracy_benchmark.py --num-samples 300 --seed 44
```

---

## 相关文档

- [校准数据指南](CALIBRATION_GUIDE.md) —— 校准数据配置、格式、经验沉淀
- [使用指南](USAGE_GUIDE.md) —— 量化/评测/部署总览
- [V100 部署指南](V100_DEPLOY_GUIDE.md) —— V100 适配详情
- [V100 服务器连接指南](V100_SERVER_GUIDE.md) —— 远程连接与容器操作
