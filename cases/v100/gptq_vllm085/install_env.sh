#!/usr/bin/env bash
# =============================================================================
# GPTQ + vLLM 0.8.5 环境安装脚本 (V100 方案 A)
#
# 功能:
#   1. 创建量化环境 venv-quant (gptqmodel 工具链)
#   2. 创建部署评测环境 vllm-venv (vLLM 0.8.5)
#   3. 验证环境
#
# 用法:
#   bash cases/v100/gptq_vllm085/install_env.sh
#
# 环境要求:
#   - CUDA 12.6 (nvcc --version 检查)
#   - Python 3.10+ (python3 --version 检查)
#   - 驱动 >= 550 (nvidia-smi 检查)
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

QUANT_VENV="/app/venv-quant"
DEPLOY_VENV="/app/vllm-venv"
PYTHON_BIN="python3"

echo "=========================================="
echo "GPTQ + vLLM 0.8.5 环境安装 (V100 方案 A)"
echo "=========================================="

# ---- 0. 前置检查 ----
echo -e "${CYAN}[0/3] 前置检查...${NC}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo -e "${RED}错误: 未找到 $PYTHON_BIN${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python: $($PYTHON_BIN --version)"

if command -v nvcc >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} CUDA: $(nvcc --version | grep 'release' | awk '{print $NF}')"
else
    echo -e "  ${YELLOW}⚠${NC} 未找到 nvcc, 请确认 CUDA 12.6 已安装"
fi

# ---- 1. 创建量化环境 venv-quant ----
echo -e "${CYAN}[1/3] 创建量化环境: ${QUANT_VENV}${NC}"
if [ ! -d "$QUANT_VENV" ]; then
    "$PYTHON_BIN" -m venv "$QUANT_VENV"
    echo -e "  ${GREEN}✓${NC} 量化环境已创建"
else
    echo -e "  ${YELLOW}⚠${NC} 量化环境已存在, 跳过创建"
fi
source "$QUANT_VENV/bin/activate"
echo -e "  ${CYAN}安装量化工具链 (gptqmodel 2.0.0)...${NC}"
pip install --upgrade pip
pip install gptqmodel==2.0.0
echo -e "  ${GREEN}✓${NC} 量化工具链安装完成"

# ---- 2. 创建部署评测环境 vllm-venv ----
echo -e "${CYAN}[2/3] 创建部署评测环境: ${DEPLOY_VENV}${NC}"
if [ ! -d "$DEPLOY_VENV" ]; then
    "$PYTHON_BIN" -m venv "$DEPLOY_VENV"
    echo -e "  ${GREEN}✓${NC} 部署环境已创建"
else
    echo -e "  ${YELLOW}⚠${NC} 部署环境已存在, 跳过创建"
fi
source "$DEPLOY_VENV/bin/activate"
echo -e "  ${CYAN}安装 vLLM 0.8.5 依赖 (见 requirements-vllm085.txt)...${NC}"
pip install --upgrade pip
# requirements 文件与本脚本同目录 (cases/v100/gptq_vllm085/)
pip install -r "$(dirname "${BASH_SOURCE[0]}")/requirements-vllm085.txt"
echo -e "  ${GREEN}✓${NC} 部署环境安装完成"

# ---- 3. 验证环境 ----
echo -e "${CYAN}[3/3] 验证环境${NC}"
source "$QUANT_VENV/bin/activate"
python -c "import gptqmodel; print(f'  gptqmodel={gptqmodel.__version__}')" 2>/dev/null || echo "  gptqmodel: N/A"
source "$DEPLOY_VENV/bin/activate"
python -c "import torch, vllm; print(f'  torch={torch.__version__}, vllm={vllm.__version__}')"

echo ""
echo "=========================================="
echo "GPTQ + vLLM 0.8.5 环境安装完成"
echo "=========================================="
echo "量化环境: source ${QUANT_VENV}/bin/activate"
echo "部署环境: source ${DEPLOY_VENV}/bin/activate"
echo ""
echo "下一步: 使用 cases/v100/gptq_vllm085/quantize.sh 进行 GPTQ 量化"
echo "        使用 cases/v100/gptq_vllm085/serve.sh 启动推理服务"
