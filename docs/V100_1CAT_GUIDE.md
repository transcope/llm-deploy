# V100 方案 B：1Cat-vLLM + AWQ 部署指南

> **硬件环境**: 8 x NVIDIA V100 32GB (共 256GB 显存)
> **方案定位**: V100 上 AWQ 量化 + 1Cat-vLLM 推理（高性能新方案）
> **并列方案**: 方案 A（GPTQ + vLLM 0.8.5）见 [V100_DEPLOY_GUIDE.md](V100_DEPLOY_GUIDE.md)
> **脚本目录**: `cases/v100/awq_1cat/`

---

## 一、AWQ 在 V100 上的支持情况

**结论：官方 vLLM 的 AWQ 不支持 V100，但 1Cat-vLLM 的 AWQ 支持 V100。**

| 推理引擎      | AWQ 对 V100 的支持 | 说明                                                         |
| ------------- | ------------------ | ------------------------------------------------------------ |
| **官方 vLLM** | ❌ 不支持           | 官方文档明确 AWQ 需要 Turing 架构（SM 7.5+），V100 是 Volta 架构（SM 7.0），不满足要求 |
| **1Cat-vLLM** | ✅ **支持**         | 专门为 V100（SM70）重写了 AWQ 内核，集成 TurboMind 派生的 SM70 算子 |
| **LMDeploy**  | ✅ 支持             | TurboMind 引擎支持 V100 上运行 AWQ/GPTQ INT4 推理            |

**关键点**：官方 vLLM 的 Marlin 内核没有 SM70 的编译版本，所以 AWQ 模型在 V100 上加载会失败。1Cat-vLLM 通过自定义的 `awq_sm70` 内核绕开了这个问题。

---

## 二、基于 CUDA 12.8 的更新方案

既然 CUDA 12.8 已安装，现在可以严格遵循 1Cat-vLLM 的官方要求进行部署。

### 📋 环境要求

| 组件        | 版本要求    | 检查/安装命令          |
| ----------- | ----------- | ---------------------- |
| **CUDA**    | 12.8        | `nvcc --version`       |
| **Python**  | 3.12        | `python3.12 --version` |
| **PyTorch** | 2.9.1+cu128 | 由 1Cat-vLLM 自动安装  |
| **驱动**    | ≥550        | `nvidia-smi` 查看      |

> 1Cat-vLLM v1.0.0 的 METADATA 精确锁定 `torch==2.9.1+cu128`。

---

### 🚀 第一步：创建虚拟环境

```bash
python3.12 -m venv /app/1cat-venv
source /app/1cat-venv/bin/activate
```

---

### ⚙️ 第二步：配置 pip 国内镜像源

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
```

---

### 📦 第三步：安装 1Cat-vLLM（自动安装 PyTorch 2.9.1+cu128）

**直接安装 1Cat-vLLM wheel，它会自动拉取正确版本的 PyTorch（2.9.1+cu128）：**

```bash
# 安装 V100 专用的 FlashAttention 内核
pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl

# 安装 1Cat-vLLM 主包
pip install https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl
```

> **注意**：1Cat-vLLM 的 wheel 依赖声明了 `torch==2.9.1`，pip 会自动从 `--index-url` 指定的源（PyPI 或镜像）下载对应版本。由于国内镜像可能没有 cu128 版本的 PyTorch，建议同时指定官方源：
> ```bash
> pip install torch==2.9.1 torchvision==0.20.1 torchaudio==2.9.1 \
>  --index-url https://download.pytorch.org/whl/cu128 \
>  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
> ```

> 如果 GitHub 下载慢，可先在本地下载 wheel 文件后上传安装。

---

### 🧪 第四步：验证环境

```bash
python -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
```

预期输出：`2.9.1+cu128 True 1.0.0`

---

### 🧠 第五步：AWQ 量化原始模型

> ⚠️ **实测修正（重要）**：必须用 **AutoAWQ** 量化，产出 **AWQ 原生格式**（`quant_method=awq`）。
> **不要用 llmcompressor** —— 它产出 `compressed-tensors` 格式（`quant_method=compressed-tensors`），
> 1Cat-vLLM 的 `--quantization awq` 无法加载，会报
> `Quantization method specified in the model config (compressed-tensors) does not match ... (awq)` 错误。
> 早期用 llmcompressor 产出的 `Mind-SLLM-Qwen3-8B-AWQ-ASYM8` 等模型均无法在 1Cat-vLLM 上部署。

使用 **AutoAWQ** 将 Qwen3-8B 量化为 AWQ 原生格式（**需在独立虚拟环境中进行，避免与 1Cat-vLLM 的 torch 版本冲突**）：

```bash
# 新建量化环境
python3.12 -m venv /app/venv-quant-awq
source /app/venv-quant-awq/bin/activate

# 安装 autoawq（产出 AWQ 原生格式）
pip install autoawq
```

执行 AWQ 量化 —— **使用项目统一量化脚本 `quantize_model.py` 的 legacy AutoAWQ 路径 + `configs/awq_4bit_v100.yaml`**（该命令已在 V100 实测跑通，产物为 `Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ`，`quant_method=awq`、非对称、`zero_point: true`）：

```bash
cd /volume/workspace/llm-deploy
export PYTORCH_ALLOC_CONF=expandable_segments:True
python llm_deploy/quantize_model.py \
    --model /app/local_models/Mind-SLLM-Qwen3-8B \
    --method awq \
    --config configs/awq_4bit_v100.yaml \
    --output /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --force-legacy-awq
```

> `--force-legacy-awq` 强制走 `quantize_awq_legacy`（AutoAWQ）路径，跳过 llmcompressor。
> 量化参数由 `configs/awq_4bit_v100.yaml` 控制：`zero_point: true`、`group_size: 128`、`w_bit: 4`、校准数据 `data/calibration/calibration_data.jsonl`（32 样本 × 2048 长度）。

> **MoE 模型特别注意**：如果是 MoE 架构（如 Qwen3 MoE），需要忽略路由门控层的量化：
> ```python
> ignore=["re:.*mlp\\.gate$", "re:.*shared_expert_gate$"]
> ```

---

### 🌐 第六步：启动 1Cat-vLLM 服务

> ⚠️ **实测修正（重要）**：必须加 `--no-enable-prefix-caching --no-enable-chunked-prefill`。
> 否则长序列评测会触发 `_flash_v100_prefill_with_prefix` 路径的共享内存超限错误
> （`RuntimeError: Shared memory limit exceeded`），导致评测失败。

```bash
source /app/1cat-venv/bin/activate

# 设置 V100 专用的 FlashAttention 后端
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100

python -m vllm.entrypoints.openai.api_server \
    --model /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --port 8000 \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill
```

> 1Cat-vLLM 官方推荐使用 `FLASH_ATTN_V100` 作为注意力后端，这是专门为 V100 优化的路径。

---

### 🧪 第七步：测试服务

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ",
        "messages": [{"role": "user", "content": "你好，请介绍一下你自己"}]
    }'
```

---

### 📊 第八步：领域精度评测

> ⚠️ **实测修正（重要）**：评测脚本必须加 `--no-thinking`，避免 Qwen3 的 thinking 内容
> 耗尽 `max_tokens` 导致回答被截断（表现为 `finish_reason=length`、输出全是 thinking 内容）。

```bash
source /app/1cat-venv/bin/activate
cd /volume/workspace/llm-deploy

python llm_deploy/benchmark_domain.py \
    --base-url http://localhost:8000 \
    --model Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
    --output results/domain_awq_autoawq.json \
    --no-thinking
```

**实测结果**（AWQ-AutoAWQ 模型，175 条领域数据）：

| 来源 | 准确率 | 平均得分 | 通过数 |
|------|--------|---------|--------|
| **math** | **88.89%** | 0.8889 | 24/27 |
| **codegen** | **80.00%** | 0.4029 | 8/10 |
| **tasks** | 50.00% | 0.4167 | 3/6 |
| **alpaca** | 36.04% | 0.3266 | 40/111 |
| **messages** | 0.00% | 0.0734 | 0/21 |
| **总体** | **42.86%** | 0.3904 | 75/175 |

---

## 三、关键提醒

| 问题                    | 说明                                                         |
| ----------------------- | ------------------------------------------------------------ |
| **PyTorch 版本**        | 必须使用 `torch==2.9.1+cu128`，不能是 2.10.0 或其他版本      |
| **AWQ 量化格式**        | 必须用 **AutoAWQ 产出 AWQ 原生格式**（`quant_method=awq`）+ **非对称**（带 zero-point）。llmcompressor 的 compressed-tensors 格式不兼容 |
| **FlashAttention 后端** | 必须设置 `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100` 才能启用 V100 专用优化 |
| **prefix caching / chunked prefill** | 必须禁用（`--no-enable-prefix-caching --no-enable-chunked-prefill`），否则长序列评测触发共享内存错误 |
| **评测禁用 thinking**   | 评测脚本加 `--no-thinking`，避免 thinking 耗尽 max_tokens |
| **环境隔离**            | 量化（AutoAWQ）和推理（1Cat-vLLM）分两个虚拟环境，避免 torch 版本冲突 |
| **V100 不支持 bfloat16** | 原模型若为 bfloat16，无法直接加载（vllm 报 `Bfloat16 is only supported on GPUs with compute capability of at least 8.0`），需转 float16。转换较慢（vllm 逐张量转换约 3.6h；transformers 加载易卡住） |

---

## 四、基线（FP16 原模型）评测说明

> 量化精度损失需与 FP16 原模型基线对比。原模型 `Mind-SLLM-Qwen3-8B` 为 **bfloat16**，
> V100（SM70）不支持 bfloat16，必须转 float16 才能部署。转换过程遇到以下问题：

| 尝试方案 | 结果 | 说明 |
|---------|------|------|
| vllm `--dtype float16` 单卡加载 | ❌ 太慢 | bfloat16→float16 逐张量转换，约 3.6h（4 分片） |
| vllm `--dtype bfloat16` 加载 | ❌ 失败 | V100 不支持 bfloat16（compute capability 7.0 < 8.0） |
| transformers `from_pretrained` 预转换 | ❌ 卡住 | CPU/GPU 转换均卡在 `Loading checkpoint shards: 0%` |
| **8 卡 V100 张量并行（`-tp 8`）** | ✅ **成功** | 每卡只加载 2GB，转换并行，**加载仅 129s** |

### ✅ 推荐方案：8 卡 V100 张量并行

```bash
source /app/1cat-venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100
python -m vllm.entrypoints.openai.api_server \
    --model /volume/models/Mind-SLLM-Qwen3-8B \
    --dtype float16 \
    --tensor-parallel-size 8 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 4096 \
    --port 8000 --trust-remote-code \
    --no-enable-prefix-caching --no-enable-chunked-prefill \
    --enforce-eager
```

### 实测精度对比（FP16 基线 vs AWQ 量化）

| 来源 | FP16 基线 | AWQ 量化 | 差异 |
|------|----------|---------|------|
| **math** | 81.48% (22/27) | 88.89% (24/27) | +7.41% |
| **codegen** | 70.00% (7/10) | 80.00% (8/10) | +10% |
| **tasks** | 33.33% (2/6) | 50.00% (3/6) | +16.67% |
| **alpaca** | 36.04% (40/111) | 36.04% (40/111) | 0% |
| **messages** | 0.00% (0/21) | 0.00% (0/21) | 0% |
| **总体** | **40.57%** (71/175) | **42.86%** (75/175) | **+2.29%** |

> **结论**：AWQ 量化模型精度与 FP16 基线相当（甚至略高），**无精度损失**。
> math/codegen 领域 AWQ 表现更优；alpaca/messages 低分是评测数据特点（开放式长答案关键词匹配难），非量化问题。

---

## 五、预期性能

| 指标             | 方案 A (GPTQ + vLLM0.8.5) | 方案 B (AWQ + 1Cat-vLLM) |
| ---------------- | ----------------------- | -------------------------- |
| **解码速度**     | ~30 tokens/s            | **~90 tokens/s**           |
| **吞吐量**       | 中                      | 高（约 3 倍提升）          |
| **CUDA 版本**    | 12.6                    | **12.8** ✅                 |
| **PyTorch 版本** | 2.6.0+cu124             | **2.9.1+cu128** ✅          |
