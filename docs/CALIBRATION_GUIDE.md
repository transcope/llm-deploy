# 校准数据指南 —— 配置 / 来源 / 格式化 / 经验沉淀

> 本文档存档量化校准数据相关的全部内容：配置字段、数据来源与结构、格式化流程、
> 各后端的消费格式差异，以及从配置注释和实测中沉淀的关键经验。
>
> 适用范围：AWQ / GPTQ / FP8 / W8A8 等所有需要校准的量化方案（BitsAndBytes NF4 无需校准）。

---

## 目录

- [1. 校准数据的作用与原理](#1-校准数据的作用与原理)
- [2. 配置字段说明](#2-配置字段说明)
- [3. 数据来源与结构](#3-数据来源与结构)
- [4. 格式化流程](#4-格式化流程)
- [5. 各后端的消费格式](#5-各后端的消费格式)
- [6. 关键经验沉淀](#6-关键经验沉淀)
- [7. 自定义校准数据](#7-自定义校准数据)

---

## 1. 校准数据的作用与原理

校准（calibration）给量化算法提供一批代表性输入，用来统计激活值分布：

- **GPTQ**：用校准数据计算每层的 Hessian 矩阵，指导权重逐列量化顺序
- **AWQ**：用校准数据统计激活的缩放因子，保护显著权重
- **SmoothQuant W8A8**：用校准数据计算平滑系数 α，迁移激活异常值到权重
- **FP8 (动态)**：`FP8_DYNAMIC` 无需校准；`FP8_BLOCK` 需校准

**核心特性**：校准数据**只在校准阶段使用一次**，不进入最终模型文件。同一份量化模型，部署时不再需要校准数据。

```
原始权重(FP16) + 校准数据 ──量化算法──▶ INT4 权重 + scale/zero
                          (一次性)        ↑ 校准数据不保留
```

---

## 2. 配置字段说明

校准参数位于 YAML 配置的 `calibration` 段：

```yaml
calibration:
  num_samples: 128                                          # 校准样本数
  dataset: "neuralmagic/LLM_compression_calibration"        # HF 数据集名
  format: "chat_template"                                    # 声明性字段(未消费)
  hf_endpoint: "https://hf-mirror.com"                      # HF 镜像
  hf_cache: "/volume/hf_cache"                              # HF 缓存目录
  hf_offline: true                                          # 强制离线
```

| 字段 | 是否被代码消费 | 默认 | 说明 |
|------|:---:|------|------|
| `num_samples` | ✅ | 128 | 校准样本数。`get_calibration_texts()` 读取 |
| `dataset` | ✅ | `neuralmagic/LLM_compression_calibration` | 数据集名，空则用内置默认文本 |
| `hf_endpoint` | ✅ | — | `setup_hf_env()` 设 `HF_ENDPOINT` 环境变量 |
| `hf_cache` | ✅ | — | `setup_hf_env()` 设 `HF_HOME` / `HF_DATASETS_CACHE` |
| `hf_offline` | ✅ | false | 设 `HF_HUB_OFFLINE=1` / `HF_DATASETS_OFFLINE=1` |
| `format` | ❌ | `chat_template` | **声明性字段，代码未实际消费**。真实格式化逻辑硬编码为 `apply_chat_template` |
	| `custom_data` | ✅ | — | **本地 JSONL 路径**。`get_calibration_texts()` 优先读取，支持 `messages`/`text` 两种字段格式 |

> ⚠️ `format` 是声明性字段，代码未实际消费；真实格式化逻辑硬编码为 `apply_chat_template`。

---

## 3. 数据来源与结构

`get_calibration_texts()`（`src/quantize_model.py`）按优先级取数据：

### 来源 0：本地自定义 JSONL（`custom_data` 非空时，最高优先级）

当 YAML 配置中指定 `calibration.custom_data` 时，优先从此路径读取本地 JSONL 文件：

```python
jsonl_path = custom_data
if not os.path.isfile(jsonl_path):
    # 支持相对于项目根目录的路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    alt_path = os.path.join(project_root, jsonl_path)
    if os.path.isfile(alt_path):
        jsonl_path = alt_path

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        messages = obj.get("messages")     # 优先 messages 字段 → list[dict]
        if messages:
            texts.append(messages)
        else:
            text = obj.get("text", "")     # 否则 text 字段 → str
            if text:
                texts.append(text)
```

JSONL 每行支持两种结构（与 HF 数据集完全相同）：

- **messages 格式**（推荐，对话级校准）：
  ```json
  {"messages": [{"role": "user", "content": "解释量子计算"}, {"role": "assistant", "content": "..."}]}
  ```
- **text 格式**（纯文本校准）：
  ```json
  {"text": "Large language models are transforming NLP."}
  ```

路径解析规则：
- 绝对路径 → 直接使用
- 相对路径 → 从项目根目录（`src/` 的父目录）尝试拼接
- 文件不存在 → 打印警告并回退到下一优先级

### 来源 1：HuggingFace 数据集（`dataset` 非空时）

读取数据集 `train` split 的前 `num_samples` 条，按样本字段自动分流：

```python
ds = load_dataset(dataset_name, split="train")
for sample in ds:
    messages = sample.get("messages")   # 优先取 messages 字段
    if messages:
        texts.append(messages)          # → 保留为 list[dict] (对话格式)
    else:
        text = sample.get("text", "")   # 否则取 text 字段
        if text:
            texts.append(text)          # → 保留为 str (纯文本)
```

数据集样本可以是两种结构之一：

**结构 A — messages 对话格式**（返回 `list[dict]`）：
```json
{
  "messages": [
    {"role": "user", "content": "解释量子计算"},
    {"role": "assistant", "content": "..."}
  ]
}
```

**结构 B — 纯文本格式**（返回 `list[str]`）：
```json
{"text": "The field of large language models is rapidly evolving."}
```

默认数据集 `neuralmagic/LLM_compression_calibration` 提供的是 messages 格式。

### 来源 2：内置默认文本（数据集加载失败时）

`DEFAULT_CALIBRATION_TEXTS`，10 条中英文短句，取前 `num_samples` 条：

```python
[
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
```

> ⚠️ 兜底用，领域覆盖窄。正式量化应优先用真实数据集。

---

## 4. 格式化流程

无论数据来自哪条路径，最终都经过**对话模板格式化**（`apply_chat_template`），转成模型实际会见到的 prompt 字符串：

```
原始数据
  │
  ├─ list[dict] (messages) ──┐
  │                          │
  └─ str (纯文本) ───────────┤
                             ▼
              tokenizer.apply_chat_template(
                  messages, tokenize=False,
                  add_generation_prompt=False
              )
                             │
                             ▼
              格式化后的字符串 (Qwen 模板示例):
              "<|im_start|>user\n解释量子计算<|im_end|>\n"
                             │
                             ▼
              HF Dataset [{"text": "..."}, ...]
                             │
                             ▼
              喂给 llmcompressor.oneshot / gptqmodel.quantize
```

关键函数：
- `format_calibration_data()`：把纯文本包成 `[{"role":"user","content":text}]` 再套模板，**失败时回退原文本**
- `to_calibration_dataset()`：包装成 `datasets.Dataset`，列名为 `text`

**为什么必须套 chat_template**：Instruct 模型的权重在「带特殊 token 的对话格式」上训练，校准时用相同格式才能统计到与真实推理一致的激活分布。直接喂裸文本，激活分布偏移，量化精度变差。

---

## 5. 各后端的消费格式

不同量化后端对格式化后数据的接受形式不同：

### llmcompressor 后端（AWQ / GPTQ / FP8 / W8A8）

```python
calib_dataset = Dataset.from_list([{"text": t} for t in formatted_texts])
oneshot(
    model=model,
    recipe=recipe,
    dataset=calib_dataset,                    # ← HF Dataset, text 列
    num_calibration_samples=len(calib_data),
)
```

必须是 `datasets.Dataset`（不能是 `list[str]` 或 `list[dict]`），因为 oneshot 内部会调 `dataset.column_names` 和 `dataset.map`。

### gptqmodel 后端（GPTQ 回退路径）

```python
model.quantize(
    formatted,           # ← List[str] (已套 chat_template 的字符串)
    batch_size=1,
    tokenizer=tokenizer, # ← 让 gptqmodel 内部自己 tokenize
)
```

传 `List[str]` + tokenizer，让 gptqmodel 内部 tokenize。**不能传裸 Tensor**（`prepare_dataset` 做 `example["input_ids"]` 对 2D tensor 会 IndexError）。

### legacy AutoAWQ 后端（DEPRECATED）

```python
model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_data)
# calib_data 是 List[str] (已格式化)
```

---

## 6. 关键经验沉淀

以下经验来自实测，散落在各配置注释中，此处统一存档：

### 6.1 `num_samples` 必须 ≥ 64

> 实测 4 样本校准在 deep layer 出现 6e5 量级 error，必须 ≥ 64。

生产配置统一用 `num_samples: 128`（覆盖常见 token 分布又不至于校准耗时过长）。冒烟测试可用 16，但绝不能低于 16。

| num_samples | 效果 |
|:-----------:|------|
| 4 | ❌ 深层 error 炸到 6e5，量化失败 |
| 16 | ⚠️ 仅用于冒烟验证链路连通 |
| 64 | ✅ 下限，勉强可用 |
| 128 | ✅ 生产推荐 |

### 6.2 Qwen3 basic-pipeline 数值稳定性

Qwen3 等新架构会触发 llmcompressor "fails layer-wise assumptions"，自动降级到 basic pipeline。basic pipeline 下 Hessian 累积在 GPU 上会放大数值误差：

- `offload_hessians: true`：把 Hessian 移到 CPU，用显存换精度（必需）
- `skip_compression_stats: true`：阻止 oneshot 末尾自动推断 `sparsity_config` 写入 config.json，否则 vLLM 加载时按稀疏模型处理，偏离 W4A16 语义

这两个参数在 `gptq_4bit_v100.yaml` 中默认开启。

### 6.3 容器内离线校准

容器环境无外网时，校准数据需预下载到缓存，并强制离线：

```yaml
calibration:
  hf_endpoint: "https://hf-mirror.com"   # 国内镜像
  hf_cache: "/volume/hf_cache"           # 预下载数据的缓存目录
  hf_offline: true                       # 强制离线, 避免任何网络 HEAD 请求超时
```

`setup_hf_env()` 必须在 import llmcompressor/datasets **之前**调用（`quantize_model.py` 在最早阶段统一设置），否则 oneshot 内部 `load_dataset` 会去访问 huggingface.co 而非镜像，无外网环境超时。

### 6.4 各方案耗时参考

| 模型 | A100 40GB | 备注 |
|------|-----------|------|
| 7B AWQ | 10-15 min | llmcompressor 后端 |
| 14B AWQ | 20-30 min | |
| 32B AWQ | 40-60 min | |
	| 7B GPTQ (gptqmodel) | 90-120 min | 单卡逐层，36 layers × (128 样本 Hessian + 7 modules) |

### 6.5 量化后 PPL 闭环验证 (--validate)

> 这是校准环节**唯一的质量红线**，用于拦截因校准数据异常（重复文本、格式错误、领域严重偏移）导致的"静默失败"。

**原理**：量化完成后自动加载 baseline（原始 FP16/BF16）和量化模型，在同一组验证文本上计算 Perplexity，对比其差异。若偏移超过阈值则发出警告。

```bash
# 量化 + 自动 PPL 验证
python src/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output ./models/Qwen2.5-7B-GPTQ \
    --validate                     # 新增：量化后自动做 PPL 验证

# 单独验证（模型已量化，补跑验证）
python src/validate_calibration.py \
    --baseline Qwen/Qwen2.5-7B-Instruct \
    --quantized ./models/Qwen2.5-7B-GPTQ \
    --quantization gptq
```

**验证输出示例**：
```
═══════════════════════════════════════════
  PPL 验证结果
═══════════════════════════════════════════
  Baseline (FP16/BF16):     ppl = 8.23
  量化模型 (gptq):          ppl = 8.87
  Delta:                     +0.64  (阈值: 5.0)
  验证: ✅ 通过
═══════════════════════════════════════════
```

**行为说明**：

| 特性 | 说明 |
|------|------|
| 默认不启用 | `--validate` 是可选开关，不影响现有量化流程 |
| 结果仅警告 | PPL 超阈值**不会阻断导出**，只打印警告，由用户自行判断 |
| 验证文本 | 内置 200 条通用中英文文本（日常对话 + 知识问答 + 领域感知），无需外网 |
| 自定义验证集 | `--val-data ./data/evaluation/eval_data.jsonl`，可从领域数据构建：`python src/build_calibration_data.py --mode eval --num-samples 100` |
| 阈值调优 | `--max-ppl-delta 3.0`，默认 5.0（经验值，覆盖大部分正常量化场景） |

**典型 delta 参考值**：

| 量化方案 | 典型 PPL delta | 评价 |
|:--------:|:--------------:|------|
| W8A8 (SmoothQuant) | +0.1 ~ +0.3 | 精度损失极小 |
| AWQ W4A16 | +0.3 ~ +1.0 | 正常范围 |
| GPTQ W4A16 (desc_act) | +0.5 ~ +2.0 | 正常范围 |
| GPTQ W4A16 (no desc_act) | +1.0 ~ +4.0 | 可接受 |
| **超过 5.0** | ⚠️ **建议检查** | 校准数据或量化参数可能有问题 |

**已知限制**：
- PPL 与业务精度（GSM8K/HellaSwag 准确率）不完全正相关，PPL 低≠任务精度高
- 验证耗时：7B 模型约 10~20 分钟（V100），可接受范围内
- 需要 `lm-eval` 已安装（项目依赖的一部分）
- Baseline 和量化模型需要能同时装进 GPU（或顺序加载，耗时长但省显存）

---

## 7. 自定义校准数据

`custom_data` 字段已在 `quantize_model.py` 的 `get_calibration_texts()` 中实现（来源 0，最高优先级）。以下介绍如何准备和配置自定义校准数据。

### 7.1 数据格式要求

JSONL 文件（每行一个 JSON 对象），支持两种字段结构：

| 格式 | 字段 | 类型 | 说明 |
|------|------|------|------|
| **messages**（推荐） | `messages` | `list[dict]` | 对话格式，需包含 `role`/`content`，走 `apply_chat_template` 格式化 |
| **text**（纯文本） | `text` | `str` | 纯文本，自动包装为单轮 user 消息再套模板 |

示例：
```jsonl
{"messages": [{"role": "user", "content": "什么是 5G 核心网 SBA 架构？"}, {"role": "assistant", "content": "SBA（Service-Based Architecture）是 5G 核心网的新型架构..."}]}
{"messages": [{"role": "user", "content": "解释 transformer 中的 attention 机制"}, {"role": "assistant", "content": "Attention 机制允许模型在生成每个词时关注输入序列的不同位置..."}]}
{"text": "The capital of France is Paris."}
```

> **为什么推荐 messages 格式**：Instruct 模型的校准需要包含完整对话上下文（system prompt + user + assistant），messages 格式能携带多轮对话结构，使校准分布更接近真实推理场景。`apply_chat_template` 会自动拼接角色标记和特殊 token。

### 7.2 构建校准数据集脚本

`src/build_calibration_data.py` 提供从领域数据源自动构建校准数据集的能力：

```bash
# 基本用法：从 data/custom_data/ 下所有领域数据源混合采样 256 条
python src/build_calibration_data.py --num-samples 256 --seed 42

# 指定输出路径
python src/build_calibration_data.py --num-samples 256 --output ./data/custom_data/calibration_data.jsonl
```

**内置数据源**（位于 `data/custom_data/`）：

| 数据源 | 权重 | 格式类型 | 领域 |
|--------|------|----------|------|
| `telecom_exam/` | 0.20 | Alpaca | 通信行业考试 |
| `comm_qa_selfinst2/` | 0.15 | messages | 通信 QA 自指令 |
| `math/` | 0.15 | 原始格式 | 数学 |
| `comm_qa_selfinst1/` | 0.10 | messages | 通信 QA 自指令 |
| `agent_sft/` | 0.10 | tasks | Agent SFT |
| `comm_qa_seed/` | 0.10 | messages | 通信 QA |
| `spec_exam/` | 0.05 | Alpaca | 专项考试 |
| `agent_general/` | 0.05 | tasks | Agent 通用 |
| `agent_iridium/` | 0.05 | messages | Agent Iridium 数据 |
| `codegen/` | 0.05 | codegen | 代码生成 |

脚本自动按权重比例采样，并做以下过滤：
- **长度过滤**：32 ~ 4096 token（按中英文 2:1 比例估算）
- **长文本比例**：>512 token 的样本占比控制在 20% ~ 60%
- **去重**：`messages` 格式按 `content` 文本去重

### 7.3 在 YAML 配置中使用

将 `custom_data` 指向构建好的 JSONL 文件：

```yaml
calibration:
  num_samples: 128
  custom_data: "data/custom_data/calibration_data.jsonl"   # 本地 JSONL 路径
  # dataset 和 hf_* 字段在 custom_data 存在时被忽略,
  # 但建议保留作为回退
  dataset: "neuralmagic/LLM_compression_calibration"
  hf_endpoint: "https://hf-mirror.com"
  hf_cache: "/volume/hf_cache"
  hf_offline: true
```

**优先级**：`custom_data` > `dataset` > `DEFAULT_CALIBRATION_TEXTS`（内置兜底）

### 7.4 完整流程示例

```bash
# 1. 准备领域数据到 data/custom_data/ 下各子目录
#    (已有 telecom_exam/, comm_qa_selfinst2/, math/ 等)

# 2. 构建校准数据集 (256 条领域混合样本)
python src/build_calibration_data.py --num-samples 256 --seed 42

# 3. 在 YAML 中配置 custom_data 并量化
python src/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output ./models/Qwen2.5-7B-GPTQ-domain
#   校准时会自动加载 data/custom_data/calibration_data.jsonl
```

### 7.5 注意事项

- **`num_samples` 同时控制**：从 JSONL 中读取的最大行数（非精确样本数，因为每行可能包含多轮对话）
- **文件编码**：必须为 UTF-8
- **路径**：支持相对于项目根目录的路径（如 `data/custom_data/calibration_data.jsonl`）
- **回退行为**：JSONL 文件不存在或读取失败时，自动回退到 HF 数据集或内置默认文本，量化不会中断

### 7.6 V100 校准数据 v2：Token 长度适配

> 首次量化 Mind-SLLM-Qwen3-8B 时遇到 CUDA OOM（第 2 条校准样本 17,330 tokens 导致 SDPA math 后端物化 35.8 GiB 注意力矩阵 > 31.7 GiB 可用显存）。解决方案是创建 **v2 校准数据集**，确保所有样本 token 长度不超过当前方案的安全阈值。

#### 背景

V100 上 PyTorch SDPA 的注意力后端选择：

| 后端 | 序列长度上限 | V100 支持 | 显存复杂度 |
|------|:----------:|:--------:|:---------:|
| FlashAttention | — | ❌ (SM 8.0+) | — |
| Memory-Efficient | ✅ 无硬上限 | ✅ | O(n) |
| Math (fallback) | ≤ 16384 tokens 正常工作 | ✅ | O(n²) |

> 当前方案未强制 SDPA 后端选择，PyTorch 在 V100 上对超长序列（>16K tokens）自动回退到 math 后端，物化完整 O(n²) 注意力矩阵。以 17,330 tokens 为例：17,330² × 2 bytes (FP16) × 2 (QK^T + softmax) ≈ 1.2 GiB 单层，但加上多头注意力 (32 heads) 的中间激活和 Hessian 矩阵，总显存需求超过 32 GiB，导致 OOM。

#### v2 数据集创建方法

在容器内执行以下步骤生成 v2 校准数据：

```bash
# 1. 用 Qwen3 tokenizer 逐条 tokenize，过滤超长样本
python3 -c "
import json, sys
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained('/app/local_models/Mind-SLLM-Qwen3-8B', trust_remote_code=True)
MAX_TOKENS = 8192

with open('data/calibration/calibration_data.jsonl') as f:
    lines = f.readlines()

kept, dropped = 0, 0
with open('data/calibration/calibration_data_v2.jsonl', 'w') as out:
    for line in lines:
        obj = json.loads(line)
        msgs = obj.get('messages', [])
        text = tk.apply_chat_template(msgs, tokenize=False) if msgs else obj.get('text', '')
        tokens = tk.encode(text)
        if len(tokens) <= MAX_TOKENS:
            out.write(line)
            kept += 1
        else:
            dropped += 1

print(f'v2: {kept} kept, {dropped} dropped (max {MAX_TOKENS} tokens)')
"
```

#### v2 数据集统计

| 指标 | v1 (原始) | v2 (过滤后) |
|------|:-------:|:----------:|
| 样本数 | 256 | **230** |
| 最大 token 长度 | 17,330 | **8,073** |
| 最小 token 长度 | 37 | 37 |
| 平均 token 长度 | ~1,850 | ~1,200 |
| 丢弃样本 | — | 26 (全为 Agent 多轮函数调用) |

#### 配置使用

YAML 中指向 v2 文件：

```yaml
calibration:
  num_samples: 230
  custom_data: "data/calibration/calibration_data_v2.jsonl"
```

> **注意**：v2 过滤掉了 26 条 Agent 多轮对话（全部超长样本），如果下游任务高度依赖 Agent 场景，应考虑改用 memory-efficient SDPA 后端（不修改方案逻辑，仅强制 PyTorch 后端选择）以支持长序列校准。

---

## 相关文档

- [使用指南](USAGE_GUIDE.md) —— 量化/评测/部署总览
- [V100 部署指南](V100_DEPLOY_GUIDE.md) —— V100 校准相关配置实例
- [A100 部署指南](A100_DEPLOY_GUIDE.md) —— A100 AWQ 量化耗时参考
