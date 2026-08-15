# V100 服务器部署专用指南（方案 A：GPTQ + vLLM 0.8.5）

> **硬件环境**: 8 x NVIDIA V100 32GB (共 256GB 显存)  
> **Docker 基础镜像**: Ubuntu 22.04 + CUDA 12.1  
> **适配说明**: V100 (Volta, SM 7.0) 有特殊限制，本指南针对性调整方案
>
> **方案选择**：本指南为 **方案 A（GPTQ + vLLM 0.8.5，稳定）**。
> 高性能新方案 **方案 B（AutoAWQ + 1Cat-vLLM）** 见 [V100_1CAT_GUIDE.md](V100_1CAT_GUIDE.md)，
> 脚本目录 `cases/v100/awq_1cat/`。两方案并列独立，按需选择。

---

## 1. V100 架构限制速查

| 特性 | V100 (SM 7.0) | 对比架构 | 影响 |
|------|--------------|---------|------|
| **FP8** | ❌ 不支持 | H100 (SM 9.0) 原生支持 | 无法使用 FP8 量化 |
| **AWQ GEMM** | ❌ 不支持 | RTX 20+/A100 (SM 75+) 支持 | AWQ 量化模型只能用 GEMV (慢) |
| **FlashAttention-2** | ❌ 不支持 | A100 (SM 8.0+) 支持 | vLLM 自动 fallback 到 xFormers |
| **GPTQ (EXL2)** | ✅ 支持 | 全架构支持 (SM 70+) | **推荐量化方案** |
| **BitsAndBytes NF4** | ✅ 支持 | 全架构支持 | 兼容性最好的量化方案 |
| **INT8 (W8A8)** | ✅ 支持 | 全架构支持 | 精度损失最小的量化方案 |
| **Tensor Core** | ✅ V1 | A100 为 V3 | 矩阵运算速度较慢 |

### 量化方案在 V100 上的选择策略

```
                    ┌─────────────────────────────────────┐
                    │     V100 量化方案决策树              │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
            追求最大吞吐                     追求最高精度
        (显存有限场景)                    (精度敏感场景)
                    │                               │
                    ▼                               ▼
        ┌───────────────────┐           ┌───────────────────┐
        │ GPTQ INT4 (EXL2)  │           │ 原始 BF16/FP16    │
        │ 显存节省 75%       │           │ 无精度损失         │
        │ 加速 ~2x          │           │ 需要更多 GPU       │
        └───────────────────┘           └───────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────┐
                    │ 折中方案:          │
                    │ BitsAndBytes NF4   │
                    │ 动态量化, 免转换    │
                    └───────────────────┘
```

---

## 2. 快速启动 (Docker)

> **从零恢复**：若 `/volume/workspace/llm-deploy/` 已被清空，需先恢复项目代码再进行后续操作。
> 详见 [FROM_SCRATCH_RUNBOOK.md 步骤 2](FROM_SCRATCH_RUNBOOK.md#2-恢复项目代码)。
> 物理服务器（非 Docker）的双 venv 重建见 [V100_SERVER_GUIDE.md 3.4 节](V100_SERVER_GUIDE.md#34-从零重建)。

### 2.0 前置依赖

从零执行量化前，需确认以下保留项存在：

| 保留项 | 路径 | 用途 | 清空时是否保留 |
|--------|------|------|:--------------:|
| 原始模型 | `/app/local_models/Mind-SLLM-Qwen3-8B` | 量化输入 | ✅ 保留 |
| HF 缓存 | `/volume/hf_cache` | 离线校准数据回退 | ✅ 保留 |
| 既有量化模型 | `/volume/models/` | 复用已量化模型 | ✅ 保留 |
| 项目代码 | `/volume/workspace/llm-deploy/` | 脚本 + 配置 | ❌ 需恢复 |
| 量化环境 | `/app/venv-quant/` | gptqmodel 工具链 | ❌ 需重建 |
| 部署环境 | `/app/vllm-venv/` | vLLM 0.8.5 推理 | ❌ 需重建 |
| 领域数据 | `data/custom_data/` | 校准数据源 | ❌ 需恢复（从 `/volume/datahub/` 复制） |

> ⚠️ **HF 缓存依赖**：`configs/gptq_4bit_v100_gptqmodel.yaml` 中 `hf_offline: true` 依赖
> `/volume/hf_cache` 预下载缓存。当 `custom_data` 不可用时，`get_calibration_texts()` 回退到
> `neuralmagic/LLM_compression_calibration` 数据集，需从此缓存加载。清空项目目录不影响此缓存
> （位于 `/volume/` 下而非项目目录内），但重建环境后需确认缓存路径正确。

### 2.1 构建镜像

```bash
cd /path/to/llm-deploy

# 构建 Docker 镜像 (约 15-20 分钟)
docker build -f docker/Dockerfile -t llm-deploy:v100 .

# 或使用 docker-compose
cd docker
docker-compose build
```

### 2.2 启动容器

```bash
cd docker

# 方式1: 使用 docker-compose (推荐)
docker-compose up -d vllm-server

# 方式2: 直接使用 docker run
docker run -d \
    --name vllm-v100 \
    --runtime nvidia \
    --gpus all \
    -p 8000:8000 \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e VLLM_ATTENTION_BACKEND=XFORMERS \
    -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/results:/app/results \
    -v $(pwd)/cache:/app/cache \
    llm-deploy:v100 \
    tail -f /dev/null

# 进入容器
docker exec -it vllm-v100 bash
```

### 2.3 使用一键部署脚本

```bash
# 进入容器后
docker exec -it vllm-v100 bash

# 查看支持的模型
./v100-deploy.sh --list

# 部署 Qwen2.5-7B (单卡, FP16)
./v100-deploy.sh qwen2.5-7b

# 部署 Qwen2.5-7B GPTQ INT4 (单卡, V100 推荐)
./v100-deploy.sh qwen2.5-7b-gptq

# 部署 Qwen2.5-7B AWQ (单卡, 更省显存但 V100 上较慢)
./v100-deploy.sh qwen2.5-7b-awq

# 部署 Qwen2.5-32B (双卡张量并行)
./v100-deploy.sh qwen2.5-32b

# 部署 DeepSeek-R1-14B
./v100-deploy.sh deepseek-r1-14b

# 多模态模型
./v100-deploy.sh qwen2.5-vl-7b
```

---

## 3. 8卡 V100 显存规划

### 3.1 显存需求速查表

| 模型 | BF16 显存 | GPTQ-INT4 显存 | AWQ-INT4 显存 | V100 部署建议 |
|------|----------|---------------|--------------|--------------|
| Qwen2.5-7B | ~14 GB | ~4 GB | ~4 GB | **单卡** |
| Qwen2.5-14B | ~28 GB | ~8 GB | ~8 GB | **单卡** |
| Qwen2.5-32B | ~64 GB | ~18 GB | ~18 GB | **单卡** (INT4) / 双卡 (BF16) |
| Qwen2.5-72B | ~144 GB | ~40 GB | ~40 GB | **双卡** (INT4) / 四卡 (BF16) |
| DeepSeek-R1-14B | ~28 GB | ~8 GB | ~8 GB | **单卡** |
| DeepSeek-R1-32B | ~64 GB | ~18 GB | ~18 GB | **单卡** (INT4) / 双卡 (BF16) |
| Qwen2.5-VL-7B | ~16 GB | ~5 GB | ~5 GB | **单卡** |
| Qwen2.5-VL-72B | ~150 GB | ~42 GB | ~42 GB | **双卡** (INT4) |

### 3.2 多模型并行部署策略

8卡 V100 (256GB 总显存) 可以同时部署多个模型：

```
┌────────────────────────────────────────────────────────────────┐
│                    8x V100 32GB 部署方案示例                     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  GPU 0-1: Qwen2.5-32B-GPTQ (18GB) + 备用/评测                  │
│  GPU 2-3: DeepSeek-R1-32B-GPTQ (18GB) + 备用                   │
│  GPU 4:   Qwen2.5-7B-GPTQ (4GB) + Qwen2.5-14B-GPTQ (8GB)      │
│  GPU 5:   Qwen2.5-VL-7B (5GB) + 其他轻量模型                   │
│  GPU 6-7: 预留用于大模型 BF16 部署或负载均衡                    │
│                                                                │
│  总显存使用: ~65GB / 256GB (约 25%, 大量余量用于 KV Cache)      │
└────────────────────────────────────────────────────────────────┘
```

---

## 4. 量化方案详解 (V100 适配版)

### 4.1 GPTQ INT4 (★ 推荐)

GPTQ 的 EXL2 kernel 支持 SM 70+，是 V100 上性能最好的 4-bit 量化方案。

**使用预量化模型 (推荐)**:

```bash
# 直接从 HuggingFace 下载预量化 GPTQ 模型
vllm serve Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --quantization gptq \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

**自行量化**:

```bash
# 使用 GPTQModel 量化 (容器内执行)
python llm_deploy/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method gptq \
    --output /app/models/Qwen2.5-7B-GPTQ \
    --w-bit 4 \
    --group-size 128

# 部署量化模型
vllm serve /app/models/Qwen2.5-7B-GPTQ \
    --quantization gptq \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

#### 4.1.1 GPTQ 双后端选择（V100/Qwen3 生产链路）

GPTQ 量化有两个后端，产出格式和 V100 兼容性不同。**V100 必须用 gptqmodel 后端**：

| 后端 | 配置文件 | 产出格式 | vLLM min_capability | V100 兼容 |
|------|----------|----------|:---:|:---:|
| llmcompressor | `configs/gptq_4bit_v100.yaml` | compressed-tensors | 80 (A100+) | ❌ V100 加载报错 |
| **gptqmodel** | `configs/gptq_4bit_v100_gptqmodel.yaml` | 标准 GPTQ | 60 | ✅ V100 走 Exllama kernel |

**为什么 V100 必须用 gptqmodel 后端**：llmcompressor 的 `GPTQModifier` 产出 compressed-tensors 格式，vLLM 加载时 W4A16 scheme 的 `get_min_capability()=80`，V100 (SM 7.0) 直接报错。gptqmodel 产出标准 GPTQ 格式 (`quant_method=gptq`)，vLLM `GPTQConfig.get_min_capability()=60`，V100 走 Exllama kernel，A100+ 走 Marlin。

**V100 生产量化命令（推荐）**：

```bash
python llm_deploy/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output /app/models/Qwen2.5-7B-GPTQ

# 部署: 注意 quantization 用 gptq (gptqmodel 后端产出标准 GPTQ 格式)
vllm serve /app/models/Qwen2.5-7B-GPTQ \
    --quantization gptq \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

**Qwen3 兼容**：gptqmodel 2.0 的 `MODEL_MAP` 不含 qwen3，`quantize_model.py` 会在 `GPTQModel.from_pretrained` 之前自动安装 `qwen3_gptq_adapter`（复用 Qwen2GPTQ 结构，`layer_type=Qwen3DecoderLayer`），无需手动处理。

> 校准数据配置（`num_samples`、离线校准等）见 [CALIBRATION_GUIDE.md](CALIBRATION_GUIDE.md)。

#### 4.1.2 llmcompressor 后端的适用场景

llmcompressor 后端（`gptq_4bit_v100.yaml`）产出的 compressed-tensors 格式**只能用于 A100+**。它的优势是对 Qwen3 等新架构的 pipeline 支持更好。如果你的部署目标是 A100/H100，可以用这个后端，部署时 `--quantization compressed-tensors`。

### 4.2 BitsAndBytes NF4 (免转换)

BitsAndBytes 提供动态量化，无需预先量化模型，加载时自动转换。适合快速验证和显存紧张场景。

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --quantization bitsandbytes \
    --load-format auto \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
```

**优点**: 无需等待量化，即开即用  
**缺点**: 首次加载较慢，推理速度不如预量化模型

### 4.3 SmoothQuant W8A8 (高精度)

如果 GPTQ INT4 的精度不能满足要求，可以使用 W8A8 SmoothQuant。

```bash
# 使用 llm-compressor 量化
python llm_deploy/quantize_model.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --method w8a8 \
    --output /app/models/Qwen2.5-7B-W8A8

# 部署
vllm serve /app/models/Qwen2.5-7B-W8A8 \
    --quantization compressed-tensors \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

### 4.4 AWQ 在 V100 上的注意事项

AWQ 的 GEMM kernel 需要 SM 75+ (Turing+)，V100 (SM 70) 只能使用较慢的 GEMV kernel。

```bash
# AWQ 在 V100 上可用但较慢
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

如果已有 AWQ 模型，可以先用着；但如果从头量化，**建议使用 GPTQ 替代**。

---

## 4.5 V100 + Qwen3 部署方案（实际验证）

> ⚠️ **vLLM 0.7.1 不支持 Qwen3 架构**：报错
> `Model architectures ['Qwen3ForCausalLM'] are not supported for now`，无法用 vLLM 部署 Qwen3 模型。
> 且 transformers 4.51.0 无法加载 gptqmodel 2.0.0 的 GPTQ 模型（要求 gptqmodel>=7.0.0）。

### 4.5.1 最新落地方案：vLLM 0.8.5（推荐）

**vLLM 0.8.5 支持 Qwen3 + CUDA 12.6 + V100**（V0 引擎 + XFORMERS），推理速度约 **30 tok/s**，
比 gptqmodel TORCH backend（2.6 tok/s）快约 **11.5 倍**。使用项目提供的 `llm_deploy/serve_vllm085.py`
（OpenAI 兼容 API）。

**环境**：`/app/vllm-venv`（vllm 0.8.5 + torch 2.6.0+cu124 + transformers 4.57.6），
依赖清单见 `cases/v100/gptq_vllm085/requirements-vllm085.txt`。

```bash
# 启动服务（后台，容器内）
docker exec -d ${V100_CONTAINER} bash -c 'nohup /app/vllm-venv/bin/python /volume/workspace/llm-deploy/llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0 > /tmp/serve_vllm085.log 2>&1 &'

# 验证
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"请用一句话介绍中国的春节。"}],"max_tokens":256}'
```

**serve_vllm085.py 关键逻辑**：
1. 用 vLLM 0.8.5 的 `LLM()` 直接加载（避免 `vllm serve` 的 multiprocessing 内存 profiling bug）
2. V100 必需参数：`VLLM_ATTENTION_BACKEND=XFORMERS` + `enforce_eager=True` + `dtype=float16`
3. 提供 `/v1/models`、`/v1/chat/completions`、`/health` 接口

**已知问题与规避**：
- **`vllm serve` 内存 profiling 失败**（`Initial used memory > currently used memory`）：
  用 `LLM()` 直接加载（uniproc executor）规避
- **单条/无 system 消息的 prompt 产生退化输出 "!!!!"**（V0 引擎 bug）：
  用带 system 消息的不同 dummy prompt 填充到 2 条，取第一条结果
- **低 max_tokens 截断 Qwen3 thinking 阶段**：强制至少 256 tokens

### 4.5.2 回退方案：gptqmodel + TORCH backend

若 vLLM 0.8.5 环境不可用，可回退到 **gptqmodel + TORCH backend** 部署（纯 torch 实现，V100 最兼容）。
使用项目提供的 `serve_gptq.py` 脚本（OpenAI 兼容 API）。

```bash
# 启动服务（后台）
docker exec -d ${V100_CONTAINER} bash -c 'nohup /app/venv-deploy/bin/python /volume/workspace/llm-deploy/cases/v100/serve_gptq.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --host 0.0.0.0 --port 8000 > /tmp/serve.log 2>&1 &'

# 验证
curl http://localhost:8000/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" --data @/tmp/test_chat.json
```

**serve_gptq.py 关键逻辑**：
1. 调用 `install_qwen3_gptq_adapter()` 注入 Qwen3 支持（gptqmodel 2.0.0 默认不识别 qwen3）
2. 用 `GPTQModel.from_quantized(..., backend=BACKEND.TORCH)` 加载（TORCH backend 兼容 V100）
3. 提供 `/v1/models`、`/v1/chat/completions`、`/health` 接口

**为什么不用其他 backend**：
- **ExllamaV2**：需 SM 8.0+（A100），V100（SM 7.0）报 `no kernel image is available`
- **ExllamaV1**：V100 可加载，但加载后 CUDA context 被污染，后续操作报错
- **TORCH**：纯 torch 实现，无自定义 CUDA kernel，**兼容性最好**（速度较慢但功能正常）

> **方案对比**：
>
> | 方案 | 环境 | 速度 | 说明 |
> |------|------|------|------|
> | **vLLM 0.8.5**（推荐） | `/app/vllm-venv` | ~30 tok/s | 支持 Qwen3，快 11.5 倍 |
> | gptqmodel TORCH（回退） | `/app/venv-deploy` | ~2.6 tok/s | 纯 torch，最兼容 |

---

## 5. vLLM 在 V100 上的最佳配置

### 5.1 环境变量 (必需)

```bash
# 必须设置 (V100 多进程启动方式)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 使用 xFormers 替代 FlashAttention-2
export VLLM_ATTENTION_BACKEND=XFORMERS

# GPU 顺序
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# 可选: NCCL 调优
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
```

### 5.2 启动参数推荐

```bash
vllm serve <MODEL> \
    --dtype float16 \                    # V100 上 float16 比 bfloat16 更稳定
    --gpu-memory-utilization 0.90 \      # 留 10% 余量
    --enable-prefix-caching \            # 前缀缓存 (RAG/多轮对话加速)
    --enable-chunked-prefill \           # 分块预填充 (降低 TTFT)
    --max-num-seqs 256 \                 # 最大并发序列
    --swap-space 16 \                    # CPU 交换空间 (GB)
    --trust-remote-code                  # Qwen/DeepSeek 必需
```

### 5.3 多卡并行配置

```bash
# 2卡张量并行 (32B 模型)
CUDA_VISIBLE_DEVICES=0,1 vllm serve Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --dtype float16 \
    --gpu-memory-utilization 0.92

# 4卡张量并行 (72B 模型 BF16)
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-72B-Instruct \
    --tensor-parallel-size 4 \
    --dtype float16 \
    --gpu-memory-utilization 0.92
```

---

## 6. 评测验证

### 6.1 精度评测（已弃用）

> ⚠️ **标准 Benchmark 精度评测（lm-eval `--tasks`）已弃用**，仅用于最初可行性验证。
> 精度评测请改用 `benchmark_domain.py` 领域精度评测（见 [评估协议](EVALUATION_PROTOCOL.md)）。
> 本节为历史遗留，仅供参考。

```bash
# 在容器内执行

# 1. 评测原始模型精度 (基准)
python llm_deploy/benchmark_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tasks gsm8k,hellaswag \
    --output /app/results/baseline

# 2. 评测量化模型精度
python llm_deploy/benchmark_eval.py \
    --model /app/models/Qwen2.5-7B-GPTQ \
    --quantization gptq \
    --tasks gsm8k,hellaswag \
    --baseline-model Qwen/Qwen2.5-7B-Instruct \
    --output /app/results/gptq
```

### 6.2 性能基准测试

```bash
# 先启动服务 (后台)
./v100-deploy.sh qwen2.5-7b-gptq &

# 等待服务就绪
sleep 30

# 性能测试
python llm_deploy/benchmark_eval.py \
    --model Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4 \
    --perf-test \
    --base-url http://localhost:8000 \
    --num-prompts 100 \
    --concurrency 10 \
    --max-tokens 512
```

### 6.3 V100 上预期的性能指标

| 模型 | 量化 | 单卡吞吐 (tok/s) | TTFT (ms) | 显存占用 |
|------|------|-----------------|-----------|---------|
| Qwen2.5-7B | BF16 | ~800 | ~120 | ~14GB |
| Qwen2.5-7B | GPTQ-INT4 | ~1400 | ~100 | ~4GB |
| Qwen2.5-32B | BF16 (TP=2) | ~600 | ~200 | ~64GB |
| Qwen2.5-32B | GPTQ-INT4 | ~1000 | ~150 | ~18GB |
| DeepSeek-R1-14B | BF16 | ~700 | ~150 | ~28GB |
| DeepSeek-R1-14B | GPTQ-INT4 | ~1200 | ~120 | ~8GB |

> 注: V100 相比 A100 吞吐量约低 30-40%，但量化后仍可满足生产需求。

---

## 7. 故障排查

### 7.1 常见问题

**问题1: `CUDA out of memory`**

```bash
# 降低显存利用率
--gpu-memory-utilization 0.85

# 或减少最大序列长度
--max-model-len 4096

# 或使用量化模型
--quantization gptq
```

**问题2: `RuntimeError: CUDA error: invalid device ordinal`**

```bash
# 检查 GPU 可见性
nvidia-smi

# 确认 CUDA_VISIBLE_DEVICES 设置正确
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

**问题3: `ValueError: Bfloat16 is only supported on GPUs with compute capability >= 8.0`**

```bash
# V100 不支持 bfloat16, 改用 float16
--dtype float16
```

**问题4: AWQ 模型加载极慢**

```bash
# V100 上 AWQ 使用 GEMV kernel, 速度较慢
# 建议改用 GPTQ 模型
# 或使用 --quantization awq --dtype float16 (必须)
```

**问题5: 多卡启动卡住**

```bash
# 确保设置了环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# 检查 NCCL
python -c "import torch; print(torch.distributed.is_nccl_available())"

# 如果 NCCL 有问题，尝试禁用 P2P
export NCCL_P2P_DISABLE=1
```

### 7.2 诊断脚本

```bash
# 检查环境
docker exec vllm-v100 python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {p.name}, {p.total_memory/1024**3:.1f}GB, SM{p.major}.{p.minor}')
"

# 测试 vLLM 加载
python -c "
from vllm import LLM
llm = LLM('Qwen/Qwen2.5-0.5B-Instruct', trust_remote_code=True)
print('vLLM 加载成功!')
"
```

---

## 8. 文件结构

```
llm-deploy/
├── docker/
│   ├── Dockerfile              # V100 适配的 Docker 镜像定义
│   ├── docker-compose.yml      # 多服务编排
│   ├── entrypoint.sh           # 容器入口脚本 (环境检查)
│   └── v100-deploy.sh          # V100 一键部署脚本 ★
├── llm_deploy/                        # Python 核心代码
│   ├── quantize_model.py       # 量化脚本 (V100: GPTQ/BitsAndBytes/W8A8)
│   ├── deploy_server.py        # 通用部署脚本
│   ├── serve_vllm085.py        # vLLM 0.8.5 部署服务 (V100+Qwen3 推荐) ★
│   ├── benchmark_eval.py       # 性能测试 (精度评测已弃用, 改用 benchmark_domain.py)
│   ├── benchmark_domain.py     # 领域精度评测 (API 模式 + vllm 0.8.5 本地) ★
│   ├── build_accuracy_benchmark.py  # 精度评测 Benchmark 数据集构建
│   ├── build_calibration_data.py    # 校准数据集构建
│   ├── validate_calibration.py      # PPL 验证
│   ├── hf_download.py              # HuggingFace 模型/数据下载
│   ├── qwen3_gptq_adapter.py       # Qwen3 GPTQ 适配器 (自动注入)
│   └── qwen3_pipeline_patch.py     # Qwen3 pipeline patch
├── cases/                      # 执行脚本
│   └── v100/
│       ├── install_quant_tools.sh   # 量化工具链安装 (Dockerfile 调用)
│       ├── activate_quant.sh        # 激活量化环境 (物理服务器双 venv)
│       ├── activate_vllm085.sh      # 激活部署环境 (vllm-venv, vLLM 0.8.5) ★
│       └── activate_deploy.sh       # 激活旧部署环境 (venv-deploy, vllm 0.7.1)
├── configs/
│   ├── gptq_4bit.yaml                   # GPTQ 基础配置 (通用)
│   ├── gptq_4bit_v100.yaml              # GPTQ V100 (llmcompressor 后端, A100+ 部署)
│   ├── gptq_4bit_v100_gptqmodel.yaml    # GPTQ V100 (gptqmodel 后端, V100 生产推荐) ★
│   └── w8a8.yaml                        # W8A8 配置
├── models/                     # 模型存放 (挂载卷)
├── results/                    # 评测结果 (挂载卷)
├── cache/                      # HuggingFace 缓存 (挂载卷)
├── bak/v0/                     # 归档的旧版本文档
│   ├── solution_report.md      # 通用方案报告 (v0 存档)
│   └── QUANTIZATION_ISSUES_LOG.md  # 量化问题日志 (v0 存档)
└── docs/
    ├── V100_DEPLOY_GUIDE.md    # 本文件
    ├── USAGE_GUIDE.md          # 使用与适配总览
    ├── CALIBRATION_GUIDE.md    # 校准数据指南
    ├── EVALUATION_PROTOCOL.md  # 评估协议
    ├── GPU_ARCHITECTURE_GUIDE.md  # GPU 架构兼容性
    └── A100_DEPLOY_GUIDE.md    # A100 部署指南
```

---

## 9. 一键部署命令汇总

```bash
# ========== 环境准备 ==========
cd /path/to/llm-deploy/docker
docker-compose build
docker-compose up -d vllm-server
docker exec -it vllm-v100 bash

# ========== 单卡部署 (7B/14B) ==========
./v100-deploy.sh qwen2.5-7b              # 7B FP16
./v100-deploy.sh qwen2.5-7b-gptq         # 7B GPTQ INT4 (V100 推荐 ★)
./v100-deploy.sh qwen2.5-7b-awq          # 7B AWQ (V100 上较慢)
./v100-deploy.sh deepseek-r1-7b          # DeepSeek 7B
./v100-deploy.sh deepseek-r1-14b         # DeepSeek 14B

# ========== 单卡部署 32B (INT4) ==========
./v100-deploy.sh qwen2.5-32b-gptq        # 32B GPTQ 单卡可跑 (V100 推荐 ★)
./v100-deploy.sh qwen2.5-32b-awq         # 32B AWQ 单卡可跑

# ========== 多卡部署 (BF16) ==========
./v100-deploy.sh qwen2.5-32b             # 32B BF16 双卡
./v100-deploy.sh qwen2.5-72b             # 72B BF16 四卡

# ========== 多模态 ==========
./v100-deploy.sh qwen2.5-vl-7b           # 图文理解

# ========== 评测 ==========
# 领域精度评测 (推荐, 见 EVALUATION_PROTOCOL.md)
python llm_deploy/benchmark_domain.py --base-url http://localhost:8000 --model <MODEL>
# [已弃用] 标准 Benchmark 精度评测 (lm-eval)
# python llm_deploy/benchmark_eval.py --model <MODEL> --tasks gsm8k,hellaswag --output /app/results/
```
