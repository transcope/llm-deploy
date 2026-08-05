# A100 单卡部署指南 —— 量化 / 部署 / 评测 端到端

> 本文档面向 **A100（Ampere, SM 8.0）单卡** 环境，覆盖从原始模型到可用推理服务的完整流程：
> 量化压缩 → 启动服务 → 精度评测 → 性能基准。
>
> 配套脚本: [`examples/07_a100_deploy.sh`](../examples/07_a100_deploy.sh)

---

## 目录

- [1. 为什么 A100 用 AWQ](#1-为什么-a100-用-awq)
- [2. 环境准备](#2-环境准备)
- [3. 端到端一键流程](#3-端到端一键流程)
- [4. 分阶段详解](#4-分阶段详解)
  - [4.1 量化（quantize）](#41-量化quantize)
  - [4.2 部署（deploy）](#42-部署deploy)
  - [4.3 评测（eval）](#43-评测eval)
  - [4.4 性能测试（perf）](#44-性能测试perf)
- [5. 量化模型快速部署运行](#5-量化模型快速部署运行)
- [6. 显存参考与模型选型](#6-显存参考与模型选型)
- [7. 常见问题](#7-常见问题)

---

## 1. 为什么 A100 用 AWQ

A100（SM 8.0）相对 V100 的关键提升，决定了量化方案的选择：

| 能力 | V100 (SM 7.0) | **A100 (SM 8.0)** | H100 (SM 9.0) |
|------|:---:|:---:|:---:|
| bfloat16 | ❌ | ✅ | ✅ |
| AWQ GEMM kernel | ❌ (仅 GEMV，慢 3-5×) | ✅ **全速** | ✅ |
| Marlin INT4 kernel | ❌ | ✅ | ✅ |
| FlashAttention-2 | ❌ | ✅ | ✅ |
| FP8 | ❌ | ❌ | ✅ |

**结论：A100 的首选量化方案是 AWQ INT4**（GEMM kernel 全速运行，显存节省 75%，精度保留 ~95%）。FP8 是 H100 独有，A100 不可用。

| 方案 | 显存节省 | 精度保留 | A100 可用 | 推荐度 |
|------|----------|----------|:---------:|:------:|
| **AWQ INT4** | 75% | ~95% | ✅ | ★★★ 首选 |
| GPTQ INT4 | 75% | ~90% | ✅ | ★★ 次选 |
| W8A8 (SmoothQuant) | 50% | ~96% | ✅ | ★ 精度敏感 |
| 原始 BF16 | 0% | 100% | ✅ | 显存充足时 |
| FP8 | 50% | ~99% | ❌ | 需 H100+ |

---

## 2. 环境准备

### 2.1 硬件要求

- **GPU**: 1× A100 40GB 或 80GB
- **CUDA**: >= 12.1
- **显存下限**: 7B 模型 >= 16GB，32B 模型 >= 40GB（AWQ 量化后单卡需 80GB）

### 2.2 初始化环境

```bash
# 一键初始化（创建虚拟环境 + 安装依赖 + 建目录）
./init

# 激活环境
source vllm-env/bin/activate          # Linux/Mac
# vllm-env\Scripts\activate           # Windows Git Bash

# 服务器 CUDA 环境安装完整 GPU 依赖
pip install -r requirements.txt
```

### 2.3 验证 GPU

```bash
nvidia-smi
# 确认显示 A100, CUDA >= 12.1, 驱动正常
```

---

## 3. 端到端一键流程

最简单的方式——一条命令完成「量化 → 评测」全流程：

```bash
# 默认量化 Qwen2.5-7B-Instruct (AWQ INT4) 并评测
./examples/07_a100_deploy.sh all
```

> **注意**：`all` 子命令执行「量化 + 精度评测」两步。精度评测走 lm-eval 直连 vLLM，**不需要外部服务**。
> 部署服务和性能测试需要单独执行（服务需常驻前台）。

全流程结束后，会提示下一步启动服务和测性能的命令。

---

## 4. 分阶段详解

推荐分阶段执行，便于逐步验证、定位问题。

### 4.1 量化（quantize）

将原始 FP16/BF16 模型压缩为 AWQ INT4 量化模型。

```bash
# 基本用法（默认 7B）
./examples/07_a100_deploy.sh quantize

# 量化指定模型
./examples/07_a100_deploy.sh quantize Qwen/Qwen2.5-14B-Instruct ./models/Qwen2.5-14B-AWQ
```

**等价手动命令**（脚本内部即调用此命令）：

```bash
python scripts/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method awq \
    --config configs/awq_4bit.yaml \
    --output ./models/Qwen2.5-7B-AWQ
```

**量化配置** `configs/awq_4bit.yaml`：

| 参数 | 值 | 说明 |
|------|-----|------|
| `w_bit` | 4 | 权重 4-bit |
| `q_group_size` | 128 | 量化分组 |
| `version` | GEMM | A100 用 GEMM kernel |
| `num_samples` | 128 | 校准样本数 |
| `ignore` | `["lm_head"]` | 不量化的层 |

**耗时参考**（A100 40GB）：
- 7B：约 10-15 分钟
- 14B：约 20-30 分钟
- 32B：约 40-60 分钟

**产出**：`./models/Qwen2.5-7B-AWQ/` 目录，包含量化权重（safetensors）+ `config.json`（含 `quantization_config`）+ tokenizer。

> 量化是纯数学运算，产出的模型文件与 GPU 架构无关，可跨硬件迁移。详见 [GPU_ARCHITECTURE_GUIDE.md](GPU_ARCHITECTURE_GUIDE.md)。

### 4.2 部署（deploy）

启动 vLLM OpenAI 兼容推理服务。

```bash
# 部署默认 7B AWQ 模型
./examples/07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ

# 自定义端口（通过环境变量）
PORT=9000 ./examples/07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ
```

服务启动后：
- API 地址：`http://localhost:8000`
- API 文档：`http://localhost:8000/docs`
- 按 `Ctrl+C` 停止

**等价手动命令**：

```bash
python scripts/deploy_server.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --dtype bfloat16 \
    --gpu-util 0.90 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --port 8000
```

**关键参数说明**：

| 参数 | A100 取值 | 原因 |
|------|-----------|------|
| `--dtype` | `bfloat16` | A100 原生支持 BF16，数值更稳 |
| `--quantization` | （自动识别） | 脚本从 `config.json` 读取，无需手填 |
| `--gpu-util` | `0.90` | A100 显存大，可激进利用 |
| `--max-model-len` | `32768` | 32K 上下文，按需调小省显存 |
| `--enable-prefix-caching` | 开 | 加速重复前缀请求 |
| `--enable-chunked-prefill` | 开 | 长上下文预填充优化 |

> `deploy_server.py` 内置硬件约束校验（`apply_hardware_constraints`）：A100 (SM 8.0) 不会被强制降级，BF16/AWQ 按传入值生效；若误传 `fp8` 会直接报错拦截。

### 4.3 评测（eval）

使用 lm-evaluation-harness 评测量化模型精度，并与原始基线模型对比精度损失。

```bash
# 评测默认 7B AWQ 模型（自动对比基线 Qwen2.5-7B-Instruct）
./examples/07_a100_deploy.sh eval ./models/Qwen2.5-7B-AWQ

# 指定基线模型
./examples/07_a100_deploy.sh eval ./models/Qwen2.5-7B-AWQ Qwen/Qwen2.5-7B-Instruct
```

**等价手动命令**：

```bash
python scripts/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --dtype bfloat16 \
    --tasks gsm8k,hellaswag \
    --baseline-model Qwen/Qwen2.5-7B-Instruct \
    --output ./results/a100_awq_comparison
```

**评测流程**：
1. 加载基线模型（FP16/BF16），跑 `gsm8k` + `hellaswag`
2. 加载量化模型（AWQ INT4），跑相同任务
3. 对比两者的 accuracy / ppl，输出精度损失百分比

**结果输出**：`./results/a100_awq_comparison/benchmark_results.json`

**精度损失预期**（AWQ INT4 正常范围）：

| 任务 | 基线 → AWQ 预期损失 | 判定 |
|------|---------------------|------|
| gsm8k | < 1% | 🟢 优秀 |
| hellaswag | < 1% | 🟢 优秀 |
| mmlu | < 2% | 🟢/🟡 良好 |

> 评测走 lm-eval 的 `vllm` 后端，会自行启动 vLLM 实例，**不需要** 先跑 `deploy`。所以 `all` 流程里量化完直接评测。

### 4.4 性能测试（perf）

测试推理服务的吞吐量、延迟、首 token 延迟（TTFT）。**需要服务已启动**（先跑 `deploy`）。

```bash
# 另开一个终端（保持 deploy 终端运行）
./examples/07_a100_deploy.sh perf

# 指定服务地址
./examples/07_a100_deploy.sh perf http://localhost:9000
```

**等价手动命令**：

```bash
python scripts/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --perf-test \
    --skip-accuracy \
    --base-url http://localhost:8000 \
    --num-prompts 100 \
    --max-tokens 256 \
    --concurrency 10 \
    --output ./results/a100_perf
```

**测试指标**：

| 指标 | 说明 |
|------|------|
| `throughput_tokens_per_sec` | 吞吐量（token/s），A100 7B AWQ 预期 ~1800 |
| `ttft_avg_seconds` | 平均首 token 延迟，预期 ~50ms |
| `latency_p50/p99_seconds` | 请求延迟分位数 |

**结果输出**：`./results/a100_perf/benchmark_results.json`

---

## 5. 量化模型快速部署运行

量化完成后，最快拿到一个可用推理服务的方式：

### 方式一：用脚本（推荐）

```bash
# 一条命令启动 OpenAI 兼容 API
./examples/07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ
```

### 方式二：vllm CLI 直接启动

```bash
vllm serve ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code
```

### 验证服务

服务启动后，另开终端测试：

```bash
# 1. 查看可用模型
curl http://localhost:8000/v1/models

# 2. 对话测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-7B-AWQ",
    "messages": [{"role": "user", "content": "你好，请介绍一下自己"}]
  }'

# 3. Python 客户端调用（OpenAI SDK 零迁移）
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
resp = client.chat.completions.create(
    model='Qwen2.5-7B-AWQ',
    messages=[{'role':'user','content':'用一句话解释量子计算'}]
)
print(resp.choices[0].message.content)
"
```

### 部署更大模型（A100 80GB 单卡 32B）

```bash
# 32B AWQ 量化后权重约 20GB, A100 80GB 单卡可容纳
# 需覆盖默认的双卡张量并行预设 (--tensor-parallel 1)
./examples/07_a100_deploy.sh quantize Qwen/Qwen2.5-32B-Instruct ./models/Qwen2.5-32B-AWQ

vllm serve ./models/Qwen2.5-32B-AWQ \
    --quantization awq \
    --dtype bfloat16 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code
```

> A100 40GB 单卡装不下 32B AWQ（权重 20GB + KV cache + 激活），需双卡张量并行：
> `--tensor-parallel-size 2`。

---

## 6. 显存参考与模型选型

AWQ INT4 量化后的显存占用（权重，不含 KV cache）：

| 模型 | 量化后权重 | A100 40GB 单卡 | A100 80GB 单卡 |
|------|-----------|:--------------:|:--------------:|
| Qwen2.5-7B | ~5 GB | ✅ | ✅ |
| Qwen2.5-14B | ~9 GB | ✅ | ✅ |
| Qwen2.5-32B | ~20 GB | ❌（需双卡 TP=2） | ✅ |
| Qwen2.5-72B | ~40 GB | ❌（需双卡） | ✅（紧张，建议 TP=2） |
| DeepSeek-R1-Distill-Qwen-14B | ~9 GB | ✅ | ✅ |

> 实际部署需预留 KV cache 与激活显存。`--max-model-len` 越大 KV cache 占用越多，可通过调小该值或降低 `--gpu-memory-utilization` 控制。

---

## 7. 常见问题

### Q1: 量化时报错 `llm-compressor` 未安装？

```bash
pip install llmcompressor
```

AWQ 量化优先走 llm-compressor 后端，未安装时会回退到 legacy AutoAWQ（已标记 DEPRECATED）。

### Q2: 部署时 `--quantization` 要不要手填？

**不需要。** `deploy_server.py` 会自动从模型目录的 `config.json` 读取 `quantization_config`。脚本里也不传该参数，靠自动识别。

### Q3: A100 上 AWQ 和 GPTQ 怎么选？

- **默认用 AWQ**：A100 的 GEMM kernel 比 GPTQ 快 20-30%。
- **用 GPTQ 的场景**：需要与 V100 共用同一份模型文件（GPTQ 全架构通用），或已存量 GPTQ 模型。

### Q4: 32B 模型 A100 40GB 单卡能跑吗？

不能。32B AWQ 权重约 20GB，加上 KV cache 和激活会超 40GB。方案：
- 升级到 A100 80GB 单卡，或
- 双卡张量并行 `--tensor-parallel-size 2`

### Q5: 量化模型能迁移到 H100 上跑吗？

能。量化模型文件与 GPU 架构无关。在 H100 上加载同一份 AWQ 模型，输出文本逐字一致，只是推理更快。若想用 H100 的 FP8，需在 H100 上重新量化。

### Q6: `all` 全流程和分阶段执行有什么区别？

- `all`：量化 → 精度评测（lm-eval 自启 vLLM，无需外部服务）。不含部署服务和性能测试，因为服务需常驻前台。
- 分阶段：可逐步验证，部署/性能测试需两个终端配合。

---

## 相关文档

- [GPU 架构兼容性指南](GPU_ARCHITECTURE_GUIDE.md) —— V100/A100/H100 跨硬件迁移
- [V100 部署指南](V100_DEPLOY_GUIDE.md) —— V100 专用（GPTQ 量化）
- [通用方案报告](solution_report.md) —— 整体方案设计
- 项目 README —— [../README.md](../README.md)
