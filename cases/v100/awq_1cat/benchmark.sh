#!/usr/bin/env bash
# =============================================================================
# 1Cat-vLLM 领域精度评测脚本 (V100 + AWQ 方案 B)
#
# 功能:
#   对已启动的 1Cat-vLLM 服务 (OpenAI 兼容 API) 运行领域精度评测
#   使用项目统一评测脚本 benchmark_domain.py 的 API 模式
#
# ⚠️ 重要 (实测验证结论):
#   - 评测前必须先启动服务 (cases/v100/awq_1cat/serve.sh), 且服务需禁用
#     prefix caching / chunked prefill (否则长序列评测触发共享内存错误)
#   - 评测脚本默认禁用 thinking (--no-thinking), 避免 max_tokens 被
#     thinking 内容耗尽导致回答被截断
#   - 领域精度评测不依赖 lm-eval, 只需 requests
#
# 用法:
#   bash cases/v100/awq_1cat/benchmark.sh
#   bash cases/v100/awq_1cat/benchmark.sh --base-url http://localhost:8000 \
#       --model Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ \
#       --output results/domain_awq_autoawq.json
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
BASE_URL="http://localhost:8000"
MODEL_NAME="Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ"
OUTPUT="results/domain_awq_autoawq.json"
NUM_SAMPLES=0          # 0 = 全部
PASS_THRESHOLD=0.35
VENV_DIR="/app/1cat-venv"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url) BASE_URL="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --num-samples) NUM_SAMPLES="$2"; shift 2 ;;
        --pass-threshold) PASS_THRESHOLD="$2"; shift 2 ;;
        --venv) VENV_DIR="$2"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "1Cat-vLLM 领域精度评测 (V100 + AWQ 方案 B)"
echo "=========================================="
echo -e "  服务地址:  ${CYAN}${BASE_URL}${NC}"
echo -e "  模型:      ${CYAN}${MODEL_NAME}${NC}"
echo -e "  输出:      ${CYAN}${OUTPUT}${NC}"
echo -e "  采样数:    ${CYAN}${NUM_SAMPLES}${NC} (0=全部)"
echo -e "  通过阈值:  ${CYAN}${PASS_THRESHOLD}${NC}"
echo "=========================================="

# ---- 1. 激活 1Cat-vLLM 环境 ----
echo -e "${CYAN}[1/2] 激活 1Cat-vLLM 环境: ${VENV_DIR}${NC}"
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}错误: 未找到 1Cat-vLLM 环境 ${VENV_DIR}${NC}"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# ---- 2. 检查服务健康 ----
echo -e "${CYAN}[2/3] 检查服务健康状态...${NC}"
if ! python -c "
import requests, sys
try:
    r = requests.get('${BASE_URL}/health', timeout=10)
    sys.exit(0 if r.status_code == 200 else 1)
except Exception:
    sys.exit(1)
"; then
    echo -e "${RED}错误: 服务不可达 ${BASE_URL}${NC}"
    echo -e "${YELLOW}请先启动服务: bash cases/v100/awq_1cat/serve.sh${NC}"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} 服务健康"

# ---- 3. 运行领域精度评测 ----
echo -e "${CYAN}[3/3] 运行领域精度评测...${NC}"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python llm_deploy/benchmark_domain.py \
    --base-url "$BASE_URL" \
    --model "$MODEL_NAME" \
    --output "$OUTPUT" \
    --num-samples "$NUM_SAMPLES" \
    --pass-threshold "$PASS_THRESHOLD" \
    --no-thinking

echo ""
echo "=========================================="
echo "评测完成"
echo "=========================================="
echo -e "  结果文件: ${GREEN}${OUTPUT}${NC}"
echo ""
