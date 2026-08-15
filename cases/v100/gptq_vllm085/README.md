# 方案 A：GPTQ + vLLM 0.8.5 (V100 稳定部署)

> 本目录为 **V100 上 GPTQ 量化 + vLLM 0.8.5 推理** 的方案 A。
> 与方案 B（AutoAWQ + 1Cat-vLLM，见 `cases/v100/awq_1cat/`）**并列独立**，按需选择。
> 详细部署指南见 [docs/V100_DEPLOY_GUIDE.md](../../docs/V100_DEPLOY_GUIDE.md)。

## 背景

vLLM 0.7.1 不支持 Qwen3 架构，V100 上部署 Qwen3 需用 **vLLM 0.8.5**（`vllm-venv`）。
量化用 **gptqmodel 后端**产出标准 GPTQ 格式（`quant_method=gptq`），V100 走 Exllama kernel。

**实测性能**：解码速度约 **30 tok/s**，比 gptqmodel TORCH backend（2.6 tok/s）快约 **11.5 倍**。

## ✅ 实测验证结论（重要）

| # | 结论 | 说明 |
|---|------|------|
| 1 | **必须用 gptqmodel 后端量化** | 产出标准 GPTQ 格式（`quant_method=gptq`），V100 走 Exllama kernel。不要用 llmcompressor（产出 compressed-tensors 格式，`min_capability=80`，V100 加载报错） |
| 2 | **V100 需要 XFORMERS 后端** | `VLLM_ATTENTION_BACKEND=XFORMERS` + `enforce_eager` + V0 engine |
| 3 | **V100 不支持 bfloat16** | 部署用 `--dtype float16` |
| 4 | **环境隔离** | 量化（venv-quant）与部署（vllm-venv）分两个虚拟环境 |

## 环境要求

| 组件 | 版本要求 | 检查命令 |
|------|---------|---------|
| CUDA | 12.6 | `nvcc --version` |
| Python | 3.10+ | `python3 --version` |
| vLLM | 0.8.5 | `vllm-venv` |
| gptqmodel | 2.0.0 | `venv-quant` |

## 快速开始

### 一键全流程

```bash
bash cases/v100/gptq_vllm085/deploy_all.sh all
```

### 分步执行（推荐，便于定位问题）

```bash
# 1. 安装环境 (venv-quant + vllm-venv)
bash cases/v100/gptq_vllm085/install_env.sh

# 2. GPTQ 量化 (gptqmodel 后端, 标准 GPTQ 格式)
bash cases/v100/gptq_vllm085/quantize.sh

# 3. 启动推理服务 (前台运行, 另开终端做测试)
bash cases/v100/gptq_vllm085/serve.sh

# 4. 测试服务
bash cases/v100/gptq_vllm085/deploy_all.sh test

# 5. 领域精度评测 (需服务已启动)
bash cases/v100/gptq_vllm085/benchmark.sh

# 6. 性能测试
bash cases/v100/gptq_vllm085/deploy_all.sh perf
```

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `install_env.sh` | 创建 `venv-quant`（gptqmodel）+ `vllm-venv`（vLLM 0.8.5）、验证环境 |
| `quantize.sh` | 用 `quantize_model.py` gptqmodel 后端产出标准 GPTQ 格式 |
| `serve.sh` | 用 `serve_vllm085.py` 启动 OpenAI 兼容 API（XFORMERS 后端） |
| `benchmark.sh` | 领域精度评测（`benchmark_domain.py` API 模式） |
| `deploy_all.sh` | 端到端一键脚本（env/quantize/serve/test/eval/perf） |

## 手动部署命令（参考）

```bash
# 量化 (gptqmodel 后端, 标准 GPTQ 格式)
source /app/venv-quant/bin/activate
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method gptq \
    --config configs/gptq_4bit_v100_gptqmodel.yaml \
    --output /volume/models/Mind-SLLM-Qwen3-8B-GPTQ

# 服务 (vLLM 0.8.5, XFORMERS 后端)
source /app/vllm-venv/bin/activate
python llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0

# 领域精度评测
source /app/vllm-venv/bin/activate
python llm_deploy/benchmark_domain.py \
    --base-url http://localhost:8000 \
    --model Mind-SLLM-Qwen3-8B-GPTQ \
    --output results/domain_gptq.json
```

## 领域精度评测结果（实测）

GPTQ 模型（`Mind-SLLM-Qwen3-8B-GPTQ`）领域精度评测结果（86 样本）：

| 指标 | 原模型 (FP16) | GPTQ 4-bit | 精度损失 |
|------|:------------:|:----------:|:--------:|
| **总体准确率** | **43.02%** | **39.53%** | **-3.49%** |

> GPTQ 4-bit 量化保留约 **92%** 基线精度（39.53/43.02），math 类任务无损失。
> 模型体积从 **16 GB → 5.8 GB**（-63%），推理速度从 2.6 tok/s（TORCH backend）提升到
> **30 tok/s**（vLLM 0.8.5，快 11.5 倍）。

## 预期性能对比

| 指标 | 方案 A (GPTQ+vLLM0.8.5) | 方案 B (AWQ+1Cat-vLLM) |
|------|------------------------|------------------------|
| 解码速度 | ~30 tokens/s | **~90 tokens/s** |
| 吞吐量 | 中 | 高（约 3 倍提升） |
| CUDA 版本 | 12.6 | **12.8** |
| PyTorch 版本 | 2.6.0+cu124 | **2.9.1+cu128** |
