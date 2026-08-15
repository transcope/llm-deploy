# 使用指南 —— 量化 / 评测 / 快速部署 + GPU 适配

> 本文档是项目「使用方式」的总览入口。读完 README 后，从这里进入实操：
> 三阶段流程（量化 → 部署 → 评测）+ V100/A100 适配 + 按硬件导航到专版指南。
>
> - 校准数据细节见 [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)
> - V100 专版见 [V100_DEPLOY_GUIDE.md](V100_DEPLOY_GUIDE.md)
> - A100 专版见 [A100_DEPLOY_GUIDE.md](A100_DEPLOY_GUIDE.md)
> - 跨硬件兼容性见 [GPU_ARCHITECTURE_GUIDE.md](GPU_ARCHITECTURE_GUIDE.md)
> - 从零执行全链路见 [FROM_SCRATCH_RUNBOOK.md](FROM_SCRATCH_RUNBOOK.md)

---

## 目录

- [1. 整体流程概览](#1-整体流程概览)
- [2. 量化使用方式](#2-量化使用方式)
  - [2.1 五种量化方案速查](#21-五种量化方案速查)
  - [2.2 按 GPU 选方案（决策树）](#22-按-gpu-选方案决策树)
  - [2.3 量化命令模板](#23-量化命令模板)
  - [2.4 量化产出与压缩比查看](#24-量化产出与压缩比查看)
- [3. 评测使用方式](#3-评测使用方式)
  - [3.1 评测命令模板](#31-评测命令模板)
  - [3.2 精度评测](#32-精度评测)
  - [3.3 性能测试](#33-性能测试)
  - [3.4 精度损失预期](#34-精度损失预期)
- [4. 快速部署运行](#4-快速部署运行)
  - [4.1 部署命令模板](#41-部署命令模板)
  - [4.2 用部署脚本](#42-用部署脚本)
  - [4.3 用 vllm CLI](#43-用-vllm-cli)
  - [4.4 验证服务](#44-验证服务)
- [5. 按硬件导航](#5-按硬件导航)

---

## 1. 整体流程概览

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  阶段1: 量化  │ ──▶ │  阶段2: 部署  │ ──▶ │  阶段3: 评测  │
│  quantize    │     │  deploy      │     │  eval / perf │
└──────────────┘     └──────────────┘     └──────────────┘
quantize_model.py     deploy_server.py      benchmark_domain.py
+ configs/*.yaml      (+ vllm)              (+ requests, API 模式)
   ↓ 产出                ↓ 产出               ↓ 产出
./models/<model>-<Q>   OpenAI 兼容 API       ./results/
```

| 阶段 | 脚本 | 核心配置 | 产出 |
|------|------|----------|------|
| 量化 | `llm_deploy/quantize_model.py` | `configs/<方案>.yaml` | `./models/<model>-<quant>/` |
| 部署 | `llm_deploy/deploy_server.py` | `configs/vllm_serve.yaml` | `http://localhost:8000` |
| 精度评测 | `llm_deploy/benchmark_domain.py` | （命令行参数） | `./results/` |
| 性能测试 | `llm_deploy/benchmark_eval.py --perf-test` | （命令行参数） | `./results/` |

> 精度评测统一使用 `benchmark_domain.py` 领域精度评测（见 [评估协议](EVALUATION_PROTOCOL.md)）；
> `benchmark_eval.py` 仅保留性能测试（`--perf-test`）功能。

---

## 2. 量化使用方式

### 2.1 五种量化方案速查

| 方案 | 配置文件 | 显存节省 | 精度保留 | 适用 GPU |
|------|----------|----------|----------|----------|
| **AWQ INT4** | `configs/awq_4bit.yaml` | 75% | ~95% | A100+（首选） |
| **GPTQ INT4** | `configs/gptq_4bit.yaml` | 75% | ~90% | 全架构（V100 通用） |
| GPTQ V100(llmcompressor) | `configs/gptq_4bit_v100.yaml` | 75% | ~90% | V100+（compressed-tensors 格式） |
| GPTQ V100(gptqmodel) | `configs/gptq_4bit_v100_gptqmodel.yaml` | 75% | ~90% | **V100 实际生产用**（标准 GPTQ 格式） |
| FP8 | `configs/fp8.yaml` | 50% | ~99% | H100+ |
| W8A8 (SmoothQuant) | `configs/w8a8.yaml` | 50% | ~96% | 全架构（精度敏感） |
| BitsAndBytes NF4 | `configs/bitsandbytes_nf4.yaml` | ~75% | 动态 | 全架构（免预量化） |

> NF4 无需预量化，部署时 `--quantization bitsandbytes` 动态加载，没有「量化」阶段。

### 2.2 按 GPU 选方案（决策树）

```
你的 GPU 是什么?
│
├─► V100 (SM 7.0)
│   ├─► 首选: GPTQ INT4 (gptqmodel 后端, 标准 GPTQ 格式) → gptq_4bit_v100_gptqmodel.yaml
│   ├─► 精度敏感: W8A8 / BNB NF4
│   └─► 详见 V100_DEPLOY_GUIDE.md
│
├─► A100 (SM 8.0)
│   ├─► 首选: AWQ INT4 (GEMM kernel) → awq_4bit.yaml
│   ├─► 次选: GPTQ INT4 (可与 V100 共用模型)
│   └─► 详见 A100_DEPLOY_GUIDE.md
│
└─► H100/H200 (SM 9.0)
    └─► 首选: FP8 → fp8.yaml
```

### 2.3 量化命令模板

```bash
# 通用模板
python llm_deploy/quantize_model.py \
    --model <原始模型ID或路径> \
    --method <awq|gptq|fp8|w8a8> \
    --config configs/<方案>.yaml \
    --output ./models/<model>-<quant>

# A100 推荐: AWQ
python llm_deploy/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method awq \
    --config configs/awq_4bit.yaml \
    --output ./models/Qwen2.5-7B-AWQ

# V100 推荐: GPTQ (gptqmodel 后端, 标准 GPTQ 格式)
python llm_deploy/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output ./models/Qwen2.5-7B-GPTQ
```

`--method` 可省略（从 `--config` 的 `quantization.method` 读取）。CLI 的 `--w-bit` / `--group-size` 可覆盖配置值。

> 量化是纯数学运算，产出的模型文件与 GPU 架构无关，可跨硬件迁移（见 GPU_ARCHITECTURE_GUIDE.md）。
> 校准数据如何选择、`num_samples` 怎么定，见 [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)。

### 2.4 量化产出与压缩比查看

量化完成后，产出物在 `--output` 指定目录，结构为：

```
./models/<model>-<quant>/
├── config.json              # 含 quantization_config 元数据 (量化方法/位宽/分组等)
├── quantize_config.json     # 量化配置回写 (参数 + calibration 段, 供溯源)
├── quant_log.csv            # 逐层量化误差 (layer, module, loss, damp, time) — gptqmodel 后端产出
├── model-*.safetensors      # 量化后权重
└── tokenizer 等文件
```

**压缩比**（量化效果指标之一）不靠评测脚本，直接量目录大小即可：

```bash
# 实测压缩比: 量化模型 vs FP16 基线
du -sh ./models/<原始模型>/          # 基线体积, 如 16G
du -sh ./models/<model>-<quant>/     # 量化后体积, 如 5.7G
# 压缩比 = 16 / 5.7 ≈ 2.8x, 节省 = (16-5.7)/16 ≈ 64%
```

| 指标 | 理论值 (2.1 速查表) | 实测值 (V100 GPTQ W4A16, Qwen3-8B) | 差异原因 |
|------|---------------------|--------------------------------------|----------|
| 显存节省 | 75% | ~64% | 理论 75% 只算**权重部分**；实测含 lm_head/嵌入层未量化 + 量化分组元数据 + 分片开销 |

> 2.1 速查表里的"显存节省"是按权重量化位宽算的**理论预期**（FP16→INT4 权重部分省 75%）。真实压缩比以量化后 `du` 实测为准——`ignore: ["lm_head"]` 跳过的层、嵌入层、量化分组元数据都会吃掉一部分空间，所以实测通常略低于理论值。
>
> 另两个量化效果指标（精度评估、推理性能）见第 3 节：精度评估用 `benchmark_domain.py`（领域精度评测），推理性能用 `benchmark_eval.py --perf-test`。

---

## 3. 评测使用方式

> ⚠️ **标准 Benchmark 精度评测（`benchmark_eval.py --tasks`，lm-eval 的 GSM8K/HellaSwag 等）已弃用**，
> 仅用于最初可行性验证。有了领域数据评测集后，精度评测统一使用 **`benchmark_domain.py` 领域精度评测**
> （见 [评估协议](EVALUATION_PROTOCOL.md)）。`benchmark_eval.py` **仅保留性能测试**（`--perf-test`）功能。
> 下文 3.1/3.2/3.4/3.5 中 `benchmark_eval.py` 的精度评测内容均为历史遗留，仅供参考，不再使用。

> **评测与部署的关系**：性能测试必须**先部署好服务**，再对 `--base-url` 发请求压测。

### 3.1 评测命令模板

```bash
# 通用模板 - 性能测试 (必须先部署服务, 见第 4 节)
python llm_deploy/benchmark_eval.py \
    --model <量化模型路径> \
    --perf-test --skip-accuracy \
    --base-url http://localhost:8000 \
    --num-prompts 100 --max-tokens 256 --concurrency 10 \
    --output ./results/perf
```

`--quantization` 本地模型可省略（自动从 `config.json` 识别）；评测脚本里量化方式用 `--quantization`（注意不是量化脚本的 `--method`）。CLI 的 `--dtype`/`--gpu-memory-utilization`/`--enforce-eager`/`--max-num-seqs`/`--max-model-len` 可按硬件覆盖。

### 3.2 精度评测（已弃用，改用 benchmark_domain.py）

> ⚠️ 本节为历史遗留。标准 Benchmark 精度评测已弃用，请改用 `benchmark_domain.py` 领域精度评测。

用 lm-evaluation-harness 评测量化模型精度，并与基线模型对比损失：

```bash
python llm_deploy/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --dtype bfloat16 \
    --tasks gsm8k,hellaswag \
    --baseline-model Qwen/Qwen2.5-7B-Instruct \
    --output ./results/awq_comparison
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--model` | 量化模型路径或 HF ID |
| `--quantization` | 量化类型（awq/gptq/fp8/compressed-tensors）；本地模型可省略，自动从 config.json 识别 |
| `--dtype` | V100 用 `float16`，A100+ 用 `bfloat16` |
| `--tasks` | 评测任务，逗号分隔（gsm8k/hellaswag/humaneval/mmlu/...） |
| `--baseline-model` | 基线模型，用于对比精度损失 |
| `--limit` | 限制样本数（快速验证用，如 `--limit 500`） |
| `--suite` | 任务套件：`math`/`code`/`reasoning`/`knowledge`/`full` |

精度评测走 lm-eval 的 `vllm` 后端，会**自行启动 vLLM 实例**，不需要先跑 deploy。

> V100 显存调参（8B Qwen3, vocab=151k）：`--gpu-memory-utilization 0.45 --enforce-eager --max-num-seqs 16`，避免 logprobs 张量 `[B,T,152k]` 爆 OOM。详见 V100_DEPLOY_GUIDE.md。

### 3.3 性能测试

测试吞吐/延迟/TTFT，**需要服务已启动**：

```bash
# 先在另一终端启动服务 (见第 4 节), 然后:
python llm_deploy/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --perf-test \
    --skip-accuracy \
    --base-url http://localhost:8000 \
    --num-prompts 100 \
    --max-tokens 256 \
    --concurrency 10 \
    --output ./results/perf
```

| 指标 | 说明 | A100 7B AWQ 预期 |
|------|------|------------------|
| `throughput_tokens_per_sec` | 吞吐量 | ~1800 tok/s |
| `ttft_avg_seconds` | 平均首 token 延迟 | ~50ms |
| `latency_p50/p99_seconds` | 请求延迟分位数 | — |

### 3.4 精度损失预期（已弃用）

> ⚠️ 本节为历史遗留（标准 Benchmark 精度评测已弃用）。领域精度评测的判定标准见 [评估协议](EVALUATION_PROTOCOL.md) 第 8 节。

| 任务 | 基线 → AWQ 预期损失 | 判定 |
|------|---------------------|------|
| gsm8k | < 1% | 🟢 优秀 |
| hellaswag | < 1% | 🟢 优秀 |
| mmlu | < 2% | 🟢/🟡 良好 |

评测脚本会输出带 🟢🟡🟠🔴 标记的损失对比表。损失 > 5% 说明量化异常，检查校准样本数和方案配置。

### 3.5 查看评测结果

评测完成后，所有产出保存在 `--output` 指定目录：

```bash
# 性能测试 (benchmark_eval.py --perf-test) 产出
ls -la ./results/perf/
```

**领域精度评测 (benchmark_domain.py)** 产出示例：

```bash
./results/domain_eval.json       # JSON 报告，含 overall / per_source / 逐条 results
```

控制台输出示例：
```
============================================================
评测完成
  总体准确率: 68.75% (132/192)
  平均得分:   0.6875
  阈值:       pass >= 0.35

  按来源:
    alpaca               : acc=75.00%  avg_score=0.7500  (30/40)
    codegen              : acc=65.62%  avg_score=0.6562  (42/64)
    math                 : acc=68.18%  avg_score=0.6818  (60/88)
```

**快捷查看压缩比**（量化效果的另一指标）：

```bash
du -sh ./models/<原始模型>/      # 基线体积
du -sh ./models/<量化模型>/       # 量化后体积
# 压缩比 = 基线/量化后
```

> 评测结果 JSON 可导入数据分析工具做进一步对比，或跨方案汇总到 `./results/_summary/`。

---

## 4. 快速部署运行

量化完成后，两种方式拿到 OpenAI 兼容推理服务。

### 4.1 部署命令模板

```bash
# 通用模板 - 用部署脚本
python llm_deploy/deploy_server.py \
    --model <量化模型路径> \
    --quantization <awq|gptq|fp8|compressed-tensors|bitsandbytes> \
    --dtype <float16|bfloat16> \
    --gpu-util <0.0-1.0> \
    --max-model-len <上下文长度> \
    --enable-prefix-caching --enable-chunked-prefill \
    --trust-remote-code \
    --host 0.0.0.0 --port 8000

# 通用模板 - 用 vllm CLI (等价)
vllm serve <量化模型路径> \
    --quantization <awq|gptq|fp8|compressed-tensors|bitsandbytes> \
    --dtype <float16|bfloat16> \
    --gpu-memory-utilization <0.0-1.0> \
    --max-model-len <上下文长度> \
    --enable-prefix-caching \
    --trust-remote-code
```

`--quantization` 本地模型可省略（`deploy_server.py` 自动从 `config.json` 识别并按 GPU 能力校验）；`--dtype` V100 必须填 `float16`，A100+ 用 `bfloat16`。多卡加 `--tensor-parallel N`，多模态加 `--multimodal`。

> `deploy_server.py` 的 `--dry-run` 可只打印等价 vllm 命令而不启动服务，便于核对参数。

### 4.2 用部署脚本

```bash
python llm_deploy/deploy_server.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --dtype bfloat16 \
    --gpu-util 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --trust-remote-code \
    --port 8000
```

`deploy_server.py` 会自动从模型 `config.json` 识别量化方式（无需手填 `--quantization`），并按 GPU 能力校验参数：
- V100 (SM 7.0)：强制 `float16`、拒绝 FP8、AWQ 警告
- A100 (SM 8.0)：不降级，BF16/AWQ 按传入值生效
- 误传 FP8 到非 H100：直接报错拦截

A100 单卡可一键跑：`./examples/07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ`

### 4.3 用 vllm CLI

```bash
vllm serve ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code
```

### 4.4 验证服务

```bash
# 查看可用模型
curl http://localhost:8000/v1/models

# 对话测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen2.5-7B-AWQ","messages":[{"role":"user","content":"你好"}]}'

# Python (OpenAI SDK 零迁移)
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
print(c.chat.completions.create(model='Qwen2.5-7B-AWQ',
    messages=[{'role':'user','content':'用一句话解释量子计算'}]).choices[0].message.content)
"
```

### 4.5 V100 + Qwen3 专用部署（vLLM 0.8.5）

> ⚠️ **vLLM 0.7.1 不支持 Qwen3**，V100 上部署 Qwen3 需用 **vLLM 0.8.5**（`vllm-venv`）。
> 用 `llm_deploy/serve_vllm085.py`（`LLM()` 直接加载，规避 `vllm serve` 的 multiprocessing bug）。

```bash
# 部署 GPTQ 量化模型 (V100 生产推荐)
source /app/vllm-venv/bin/activate
python llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0

# 部署 FP16 原模型
python llm_deploy/serve_vllm085.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B --port 8000 --gpu 0
```

> V100 上 vLLM 0.8.5 推理速度约 **30 tok/s**，比 gptqmodel TORCH backend（2.6 tok/s）快约 **11.5 倍**。
> 详细方案对比见 [V100_DEPLOY_GUIDE.md 4.5 节](V100_DEPLOY_GUIDE.md#45-v100--qwen3-部署方案实际验证)。

---

## 5. 按硬件导航

| 你的硬件 | 推荐方案 | 去哪看详细指南 |
|----------|----------|----------------|
| **V100 (SM 7.0)** | GPTQ INT4 (gptqmodel 后端) | [V100_DEPLOY_GUIDE.md](V100_DEPLOY_GUIDE.md) |
| **A100 (SM 8.0)** | AWQ INT4 (GEMM kernel) | [A100_DEPLOY_GUIDE.md](A100_DEPLOY_GUIDE.md) |
| **H100 (SM 9.0)** | FP8 | README + `configs/fp8.yaml` |
| 跨硬件迁移 | 模型文件通用 | [GPU_ARCHITECTURE_GUIDE.md](GPU_ARCHITECTURE_GUIDE.md) |
| 校准数据 | 任何方案都涉及 | [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md) |

---

## 相关文档

- [从零执行操作手册](FROM_SCRATCH_RUNBOOK.md) —— 清空环境后从零完成压缩→部署→评估全链路
- [校准数据指南](CALIBRATION_GUIDE.md) —— 校准样本数、数据格式、离线校准、自定义校准集
- [V100 部署指南](V100_DEPLOY_GUIDE.md) —— V100 专版（GPTQ 量化、显存调参、Docker）
- [A100 部署指南](A100_DEPLOY_GUIDE.md) —— A100 单卡端到端（AWQ 量化、一键脚本）
- [GPU 架构兼容性指南](GPU_ARCHITECTURE_GUIDE.md) —— V100/A100/H100 跨硬件迁移
- [通用方案报告](solution_report.md) —— 整体方案设计
