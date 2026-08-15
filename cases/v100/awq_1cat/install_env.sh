#!/usr/bin/env bash
# =============================================================================
# 1Cat-vLLM 环境安装脚本 (V100 + AWQ 方案 B)
#
# 功能:
#   1. 创建 1cat-venv 虚拟环境 (Python 3.12)
#   2. 配置 pip 国内镜像源 (清华)
#   3. 安装 V100 专用 flash_attn_v100 内核
#   4. 安装 1Cat-vLLM v1.0.0 (自动拉取 torch==2.9.1+cu128)
#   5. 验证环境
#
# 用法:
#   bash cases/v100/awq_1cat/install_env.sh
#   bash cases/v100/awq_1cat/install_env.sh --skip-torch   # 跳过显式 torch 安装
#
# 环境要求:
#   - CUDA 12.8 (nvcc --version 检查)
#   - Python 3.12 (python3.12 --version 检查)
#   - 驱动 >= 550 (nvidia-smi 检查)
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

VENV_DIR="/app/1cat-venv"
PYTHON_BIN="python3.12"

# 1Cat-vLLM v1.0.0 wheel 下载地址
FLASH_ATTN_WHEEL="https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/flash_attn_v100-1.0.0-cp312-cp312-linux_x86_64.whl"
VLLM_WHEEL="https://github.com/1CatAI/1Cat-vLLM/releases/download/v1.0.0/vllm-1.0.0-cp312-cp312-linux_x86_64.whl"

# 是否显式安装 torch (默认 true; 若 wheel 已自动拉取可跳过)
SKIP_TORCH=false
for arg in "$@"; do
    case "$arg" in
        --skip-torch) SKIP_TORCH=true ;;
        *) echo -e "${RED}未知参数: $arg${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "1Cat-vLLM 环境安装 (V100 + AWQ 方案 B)"
echo "=========================================="

# ---- 0. 前置检查 ----
echo -e "${CYAN}[0/6] 前置检查...${NC}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo -e "${RED}错误: 未找到 $PYTHON_BIN, 请先安装 Python 3.12${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Python: $($PYTHON_BIN --version)"

if command -v nvcc >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} CUDA: $(nvcc --version | grep 'release' | awk '{print $NF}')"
else
    echo -e "  ${YELLOW}⚠${NC} 未找到 nvcc, 请确认 CUDA 12.8 已安装"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} 驱动: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
else
    echo -e "  ${YELLOW}⚠${NC} 未找到 nvidia-smi"
fi

# ---- 1. 创建虚拟环境 ----
echo -e "${CYAN}[1/6] 创建虚拟环境: ${VENV_DIR}${NC}"
if [ ! -d "$VENV_DIR" ]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    echo -e "  ${GREEN}✓${NC} 虚拟环境已创建"
else
    echo -e "  ${YELLOW}⚠${NC} 虚拟环境已存在, 跳过创建"
fi

source "$VENV_DIR/bin/activate"

# ---- 2. 配置 pip 镜像源 ----
echo -e "${CYAN}[2/6] 配置 pip 国内镜像源 (清华)${NC}"
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
echo -e "  ${GREEN}✓${NC} pip 镜像源已配置"

# ---- 3. 升级 pip ----
echo -e "${CYAN}[3/6] 升级 pip/setuptools/wheel${NC}"
pip install --upgrade pip setuptools wheel

# ---- 4. 安装 PyTorch 2.9.1+cu128 (可选) ----
if [ "$SKIP_TORCH" = false ]; then
    echo -e "${CYAN}[4/6] 安装 PyTorch 2.9.1+cu128${NC}"
    echo -e "  ${YELLOW}提示: 1Cat-vLLM wheel 会自动拉取 torch==2.9.1, 此处显式安装确保 cu128 版本${NC}"
    pip install torch==2.9.1 torchvision==0.20.1 torchaudio==2.9.1 \
        --index-url https://download.pytorch.org/whl/cu128 \
        --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple
else
    echo -e "${CYAN}[4/6] 跳过显式 torch 安装 (由 1Cat-vLLM wheel 自动拉取)${NC}"
fi

# ---- 5. 安装 1Cat-vLLM ----
echo -e "${CYAN}[5/6] 安装 1Cat-vLLM v1.0.0${NC}"
echo -e "  ${YELLOW}提示: 若 GitHub 下载慢, 可先本地下载 wheel 后改为本地路径安装${NC}"

echo -e "  ${CYAN}安装 flash_attn_v100 (V100 专用 FlashAttention 内核)...${NC}"
pip install "$FLASH_ATTN_WHEEL"

echo -e "  ${CYAN}安装 vllm-1.0.0 (1Cat-vLLM 主包)...${NC}"
pip install "$VLLM_WHEEL"

# ---- 6. 验证环境 ----
echo -e "${CYAN}[6/6] 验证环境${NC}"
python -c "import torch, vllm; print(f'  torch={torch.__version__}, cuda_available={torch.cuda.is_available()}, vllm={vllm.__version__}')"

echo ""
echo "=========================================="
echo "1Cat-vLLM 环境安装完成"
echo "=========================================="
echo "激活命令: source ${VENV_DIR}/bin/activate"
echo "预期输出: torch=2.9.1+cu128, cuda_available=True, vllm=1.0.0"
echo ""
echo "下一步: 使用 cases/v100/awq_1cat/quantize.sh 进行 AWQ 量化"
echo "        使用 cases/v100/awq_1cat/serve.sh 启动推理服务"
