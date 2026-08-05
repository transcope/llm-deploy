#!/usr/bin/env bash
# =============================================================================
# V100 量化工具链手动/可选安装脚本
# 用途:
#   - Dockerfile 中通过 INSTALL_QUANT_TOOLS=true 自动调用
#   - 也可在运行中的容器内手动执行: bash /app/cases/v100/install_quant_tools.sh
#
# 版本说明:
#   该脚本安装的版本与 vllm==0.7.1 / torch==2.5.1 保持一致:
#   - llmcompressor==0.4.0
#   - compressed-tensors==0.9.0
#   - gptqmodel==2.0.0
#   - bitsandbytes>=0.45.0
#   - auto-gptq (legacy, 可选)
# =============================================================================
set -e

# 大体积包网络容错
export PIP_DEFAULT_TIMEOUT=300
export PIP_RETRIES=5

echo "=========================================="
echo "安装 V100 量化工具链"
echo "=========================================="

# GPTQModel (AutoGPTQ 继任者, 支持 EXL2)
# 新版 gptqmodel 要求 torch>=2.8.0, 所以锁定 2.0.0
echo "[1/5] 安装 GPTQModel 2.0.0 (可选)..."
pip install --no-cache-dir gptqmodel==2.0.0 || echo "警告: GPTQModel 安装失败, 已跳过"

# AutoGPTQ (legacy, 编译可能较久)
echo "[2/5] 安装 AutoGPTQ (legacy, 可选)..."
pip install --no-cache-dir auto-gptq --no-build-isolation || echo "警告: AutoGPTQ 安装失败, 已跳过"

# BitsAndBytes (NF4 动态量化, 兼容性最好)
echo "[3/5] 安装 BitsAndBytes..."
pip install --no-cache-dir bitsandbytes>=0.45.0

# llm-compressor (SmoothQuant W8A8 / AWQ / FP8)
# 锁定 0.4.0: 唯一与 vllm==0.7.1 共用 compressed-tensors==0.9.0 的版本
echo "[4/5] 安装 llm-compressor 0.4.0 + compressed-tensors 0.9.0..."
pip install --no-cache-dir llmcompressor==0.4.0 compressed-tensors==0.9.0

# 验证
echo "[5/5] 验证安装..."
python -c "import compressed_tensors; print(f'compressed-tensors: OK')" \
    && python -c "from llmcompressor.modifiers.quantization import QuantizationModifier; print('llmcompressor: OK')" \
    && python -c "import bitsandbytes; print('BitsAndBytes: OK')" \
    && (python -c "import gptqmodel; print('GPTQModel: OK')" 2>/dev/null || echo "GPTQModel: 未安装或可选") \
    && (python -c "import auto_gptq; print('AutoGPTQ: OK')" 2>/dev/null || echo "AutoGPTQ: 未安装或可选")

echo "=========================================="
echo "量化工具链安装完成"
echo "=========================================="
