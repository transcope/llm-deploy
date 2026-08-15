#!/usr/bin/env bash
# =============================================================================
# 1Cat-vLLM 推理服务启动脚本 (V100 + AWQ 方案 B)
#
# 功能:
#   使用 1Cat-vLLM 启动 OpenAI 兼容 API 服务, 加载 AWQ 量化模型
#   启用 V100 专用 FlashAttention 后端 (FLASH_ATTN_V100)
#
# ⚠️ 重要 (实测验证结论):
#   - 必须禁用 prefix caching 和 chunked prefill (--no-enable-prefix-caching
#     --no-enable-chunked-prefill), 否则长序列评测会触发
#     _flash_v100_prefill_with_prefix 路径的共享内存超限错误
#     (RuntimeError: Shared memory limit exceeded)
#   - 模型必须是 AWQ 原生格式 (quant_method=awq, AutoAWQ 产出),
#     不能用 llmcompressor 的 compressed-tensors 格式
#
# 用法:
#   bash cases/v100/awq_1cat/serve.sh
#   bash cases/v100/awq_1cat/serve.sh --model /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
#       --port 8000 --gpu-util 0.9 --max-len 4096
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数 (实测跑通的 AutoAWQ 产物, AWQ 原生格式)
MODEL_PATH="/volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ"
PORT=8000
GPU_UTIL=0.9
MAX_LEN=4096
HOST="0.0.0.0"
VENV_DIR="/app/1cat-venv"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "1Cat-vLLM 推理服务 (V100 + AWQ 方案 B)"
echo "=========================================="
echo -e "  模型:     ${CYAN}${MODEL_PATH}${NC}"
echo -e "  端口:     ${CYAN}${PORT}${NC}"
echo -e "  GPU利用率: ${CYAN}${GPU_UTIL}${NC}"
echo -e "  最大长度: ${CYAN}${MAX_LEN}${NC}"
echo "=========================================="

# ---- 1. 激活 1Cat-vLLM 环境 ----
echo -e "${CYAN}[1/2] 激活 1Cat-vLLM 环境: ${VENV_DIR}${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}错误: 未找到 1Cat-vLLM 环境 ${VENV_DIR}${NC}"
    echo -e "${YELLOW}请先运行: bash cases/v100/awq_1cat/install_env.sh${NC}"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ---- 2. 启动服务 ----
echo -e "${CYAN}[2/2] 启动 1Cat-vLLM 服务...${NC}"

# 设置 V100 专用 FlashAttention 后端 (1Cat-vLLM 官方推荐)
export VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100

echo -e "  ${GREEN}✓${NC} VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100"
echo -e "  ${GREEN}✓${NC} 服务地址: http://${HOST}:${PORT}"
echo -e "  ${GREEN}✓${NC} API 文档: http://${HOST}:${PORT}/docs"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --host "$HOST" \
    --port "$PORT" \
    --trust-remote-code \
    --no-enable-prefix-caching \
    --no-enable-chunked-prefill

echo ""
echo "=========================================="
echo "服务已停止"
echo "=========================================="
