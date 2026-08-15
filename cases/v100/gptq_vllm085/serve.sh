#!/usr/bin/env bash
# =============================================================================
# vLLM 0.8.5 推理服务启动脚本 (V100 方案 A, GPTQ)
#
# 功能:
#   使用 serve_vllm085.py 启动 OpenAI 兼容 API 服务, 加载 GPTQ 量化模型
#   (LLM() 直接加载, 规避 vllm serve 的 multiprocessing 内存 profiling bug)
#
# ⚠️ 重要 (实测验证结论):
#   - V100 需要 XFORMERS attention backend + enforce_eager + V0 engine
#   - 模型必须是标准 GPTQ 格式 (quant_method=gptq, gptqmodel 后端产出)
#
# 用法:
#   bash cases/v100/gptq_vllm085/serve.sh
#   bash cases/v100/gptq_vllm085/serve.sh --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
#       --port 8000 --gpu 0
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
MODEL_PATH="/volume/models/Mind-SLLM-Qwen3-8B-GPTQ"
PORT=8000
GPU=0
GPU_UTIL=0.9
MAX_LEN=4096
VENV_DIR="/app/vllm-venv"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "vLLM 0.8.5 推理服务 (V100 方案 A, GPTQ)"
echo "=========================================="
echo -e "  模型:     ${CYAN}${MODEL_PATH}${NC}"
echo -e "  端口:     ${CYAN}${PORT}${NC}"
echo -e "  GPU:      ${CYAN}${GPU}${NC}"
echo "=========================================="

# ---- 1. 激活 vLLM 0.8.5 环境 ----
echo -e "${CYAN}[1/2] 激活 vLLM 0.8.5 环境: ${VENV_DIR}${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}错误: 未找到 vLLM 0.8.5 环境 ${VENV_DIR}${NC}"
    echo -e "${YELLOW}请先运行: bash cases/v100/gptq_vllm085/install_env.sh${NC}"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ---- 2. 启动服务 ----
echo -e "${CYAN}[2/2] 启动 vLLM 0.8.5 服务...${NC}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

echo -e "  ${GREEN}✓${NC} 服务地址: http://localhost:${PORT}"
echo ""

python llm_deploy/serve_vllm085.py \
    --model "$MODEL_PATH" \
    --quantization gptq \
    --port "$PORT" \
    --gpu "$GPU" \
    --gpu-util "$GPU_UTIL" \
    --max-model-len "$MAX_LEN"

echo ""
echo "=========================================="
echo "服务已停止"
echo "=========================================="
