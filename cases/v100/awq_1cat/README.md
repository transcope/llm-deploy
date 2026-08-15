# 方案 B：1Cat-vLLM + AWQ (V100 优化部署)

> 本目录为 **V100 上 AWQ 量化 + 1Cat-vLLM 推理** 的方案 B。
> 与方案 A（GPTQ + vLLM 0.8.5，见 `cases/v100/gptq_vllm085/`）**并列独立**，按需选择。
> 方案设计文档见 [docs/V100_1CAT_GUIDE.md](../../docs/V100_1CAT_GUIDE.md)。

## 背景

官方 vLLM 的 AWQ **不支持 V100**（Marlin 内核无 SM70 编译版本），因此 V100 上 AWQ 模型加载会失败。
**1Cat-vLLM** 专门为 V100（SM70）重写了 AWQ 内核（`awq_sm70`），集成 TurboMind 派生的 SM70 算子，
从而可以在 V100 上运行 AWQ 推理。

**预期性能提升**：解码速度从 ~2.6 tok/s（GPTQ+TORCH）提升到 **~90 tok/s**（约 35 倍）。

## ✅ 实测验证结论（重要）

本方案已在 V100 实测跑通，以下是**验证过程中发现的关键结论**，务必遵守：

| # | 结论 | 说明 |
|---|------|------|
| 1 | **必须用 AutoAWQ 量化** | 1Cat-vLLM 的 SM70 内核只支持 **AWQ 原生格式**（`quant_method=awq`，权重键 `qweight/qzeros/scales`）。**不要用 llmcompressor**（产出 `compressed-tensors` 格式，`--quantization awq` 无法加载，报 "Quantization method ... does not match" 错误） |
| 2 | **必须禁用 prefix caching / chunked prefill** | 启动服务加 `--no-enable-prefix-caching --no-enable-chunked-prefill`，否则长序列评测触发 `_flash_v100_prefill_with_prefix` 路径的共享内存超限错误（`RuntimeError: Shared memory limit exceeded`） |
| 3 | **评测需禁用 thinking** | 评测脚本加 `--no-thinking`，避免 Qwen3 的 thinking 内容耗尽 `max_tokens` 导致回答被截断 |
| 4 | **V100 不支持 bfloat16** | FP16 基线模型（bfloat16 原模型）无法直接加载，需转 float16（转换慢，见 [docs/V100_1CAT_GUIDE.md](../../docs/V100_1CAT_GUIDE.md) 的基线说明） |
| 5 | **环境隔离** | 量化（AutoAWQ）与推理（1Cat-vLLM）分两个虚拟环境，避免 torch 版本冲突 |

## 环境要求

| 组件 | 版本要求 | 检查命令 |
|------|---------|---------|
| CUDA | 12.8 | `nvcc --version` |
| Python | 3.12 | `python3.12 --version` |
| PyTorch | 2.9.1+cu128 | 由 1Cat-vLLM 自动安装 |
| 驱动 | ≥550 | `nvidia-smi` |

> 1Cat-vLLM v1.0.0 精确锁定 `torch==2.9.1+cu128`。

## 快速开始

### 一键全流程

```bash
bash cases/v100/awq_1cat/deploy_all.sh all
```

### 分步执行（推荐，便于定位问题）

```bash
# 1. 安装 1Cat-vLLM 环境
bash cases/v100/awq_1cat/install_env.sh

# 2. AWQ 量化 (AutoAWQ, 产出 AWQ 原生格式)
bash cases/v100/awq_1cat/quantize.sh

# 3. 启动推理服务 (前台运行, 另开终端做测试)
bash cases/v100/awq_1cat/serve.sh

# 4. 测试服务
bash cases/v100/awq_1cat/deploy_all.sh test

# 5. 领域精度评测 (需服务已启动)
bash cases/v100/awq_1cat/benchmark.sh

# 6. 性能测试
bash cases/v100/awq_1cat/deploy_all.sh perf
```

## 脚本说明

| 脚本 | 功能 |
|------|------|
| `install_env.sh` | 创建 `1cat-venv`、配置清华镜像、安装 `flash_attn_v100` + `vllm-1.0.0`、验证环境 |
| `quantize.sh` | 创建 `venv-quant-awq`、安装 autoawq、用 `quantize_model.py --force-legacy-awq` 产出 AWQ 原生格式 |
| `serve.sh` | 用 1Cat-vLLM 启动 OpenAI 兼容 API（`FLASH_ATTN_V100` 后端 + awq + 禁用 prefix caching/chunked prefill） |
| `benchmark.sh` | 领域精度评测（`benchmark_domain.py` API 模式，禁用 thinking） |
| `deploy_all.sh` | 端到端一键脚本（env/quantize/serve/test/eval/perf） |

## 关键注意事项

| 问题 | 说明 |
|------|------|
| **PyTorch 版本** | 必须 `torch==2.9.1+cu128`，不能是 2.10.0 或其他版本 |
| **AWQ 量化格式** | 必须 **AutoAWQ 产出的 AWQ 原生格式**（`quant_method=awq`）+ 非对称（带 zero-point）。llmcompressor 的 compressed-tensors 格式不兼容 |
| **FlashAttention 后端** | 必须设 `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100` 才能启用 V100 专用优化 |
| **prefix caching / chunked prefill** | 必须禁用（`--no-enable-prefix-caching --no-enable-chunked-prefill`），否则长序列评测触发共享内存错误 |
| **环境隔离** | 量化（AutoAWQ）与推理（1Cat-vLLM）分两个虚拟环境，避免 torch 版本冲突 |
| **MoE 模型** | 若是 MoE 架构，需忽略路由门控层量化：`ignore=["re:.*mlp\\.gate$", "re:.*shared_expert_gate$"]` |

## 手动部署命令（参考）

```bash
# 环境
python3.12 -m venv /app/1cat-venv
source /app/1cat-venv/bin/activate
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl
pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl

# 量化 (独立环境, AutoAWQ 产出 AWQ 原生格式, 已在 V100 实测跑通)
python3.12 -m venv /app/venv-quant-awq
source /app/venv-quant-awq/bin/activate
pip install autoawq
export PYTORCH_ALLOC_CONF=expandable_segments:True
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method awq \
    --config configs/awq_4bit_v100.yaml \
    --output /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --force-legacy-awq

# 服务 (禁用 prefix caching / chunked prefill)
source /app/1cat-venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100
python -m vllm.entrypoints.openai.api_server \
    --model /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --quantization awq --dtype float16 \
    --gpu-memory-utilization 0.9 --max-model-len 4096 \
    --port 8000 --trust-remote-code \
    --no-enable-prefix-caching --no-enable-chunked-prefill

# 领域精度评测 (禁用 thinking)
source /app/1cat-venv/bin/activate
python llm_deploy/benchmark_domain.py \
    --base-url http://localhost:8000 \
    --model Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --output results/domain_awq_autoawq.json \
    --no-thinking
```

## 测试

```bash
# 模型列表
curl http://localhost:8000/v1/models

# 对话
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ","messages":[{"role":"user","content":"你好"}]}'
```

## 领域精度评测结果（实测）

AWQ-AutoAWQ 模型（`Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ`）领域精度评测结果：

| 来源 | 准确率 | 平均得分 | 通过数 |
|------|--------|---------|--------|
| **math** | **88.89%** | 0.8889 | 24/27 |
| **codegen** | **80.00%** | 0.4029 | 8/10 |
| **tasks** | 50.00% | 0.4167 | 3/6 |
| **alpaca** | 36.04% | 0.3266 | 40/111 |
| **messages** | 0.00% | 0.0734 | 0/21 |
| **总体** | **42.86%** | 0.3904 | 75/175 |

> math/codegen 表现优秀；alpaca/messages 偏低（开放式长答案，关键词匹配较难）。
>
> **基线（FP16 原模型）对比**：原模型为 bfloat16，V100 不支持 bfloat16，需转 float16。
> 单卡加载转换极慢（约 3.6h），**用 8 卡 V100 张量并行（`-tp 8`）解决**，加载仅 129s。
> 实测对比：**AWQ 量化 42.86% vs FP16 基线 40.57%**，AWQ 无精度损失（甚至略高）。
> 详见 [docs/V100_1CAT_GUIDE.md](../../docs/V100_1CAT_GUIDE.md) 的基线说明。

## 预期性能对比

| 指标 | 方案 A (GPTQ+vLLM0.8.5) | 方案 B (AWQ+1Cat-vLLM) |
|------|------------------------|------------------------|
| 解码速度 | ~30 tokens/s | **~90 tokens/s** |
| 吞吐量 | 中 | 高（约 3 倍提升） |
| CUDA 版本 | 12.6 | **12.8** |
| PyTorch 版本 | 2.6.0+cu124 | **2.9.1+cu128** |
