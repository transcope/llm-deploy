#!/usr/bin/env bash
# =============================================================================
# 1Cat-vLLM 端到端一键部署脚本 (V100 + AWQ 方案 B)
#
# 全流程: 环境安装 → AWQ 量化 → 启动服务 → 测试验证 → 领域精度评测
#
# 用法:
#   bash cases/v100/awq_1cat/deploy_all.sh all          # 全流程 (环境+量化+部署+测试)
#   bash cases/v100/awq_1cat/deploy_all.sh env          # 仅安装环境
#   bash cases/v100/awq_1cat/deploy_all.sh quantize     # 仅 AWQ 量化 (AutoAWQ)
#   bash cases/v100/awq_1cat/deploy_all.sh serve        # 仅启动服务 (前台)
#   bash cases/v100/awq_1cat/deploy_all.sh test         # 仅测试服务 (需服务已启动)
#   bash cases/v100/awq_1cat/deploy_all.sh eval         # 领域精度评测 (需服务已启动)
#   bash cases/v100/awq_1cat/deploy_all.sh perf         # 性能测试 (需服务已启动)
#
# 注意: 量化用 AutoAWQ 产出 AWQ 原生格式 (quant_method=awq), 1Cat-vLLM SM70 内核所需。
#       llmcompressor 的 compressed-tensors 格式不兼容, 不要使用。
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCHEME_DIR="$SCRIPT_DIR"

# 默认参数 (输出目录与实测跑通的 AutoAWQ 产物一致)
MODEL_PATH="/app/local_models/Mind-SLLM-Qwen3-8B"
OUTPUT_DIR="/volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ"
PORT=8000
GPU_UTIL=0.9
MAX_LEN=4096

# 解析参数
ACTION="${1:-all}"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-util) GPU_UTIL="$2"; shift 2 ;;
        --max-len) MAX_LEN="$2"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "1Cat-vLLM 端到端部署 (V100 + AWQ 方案 B)"
echo "=========================================="
echo -e "  动作:     ${CYAN}${ACTION}${NC}"
echo -e "  模型:     ${CYAN}${MODEL_PATH}${NC}"
echo -e "  输出:     ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  端口:     ${CYAN}${PORT}${NC}"
echo "=========================================="

# ---- 1. 安装环境 ----
if [[ "$ACTION" == "all" || "$ACTION" == "env" ]]; then
    echo ""
    echo -e "${GREEN}>>> [1/4] 安装 1Cat-vLLM 环境${NC}"
    bash "$SCHEME_DIR/install_env.sh"
fi

# ---- 2. AWQ 量化 ----
if [[ "$ACTION" == "all" || "$ACTION" == "quantize" ]]; then
    echo ""
    echo -e "${GREEN}>>> [2/4] AWQ 量化${NC}"
    bash "$SCHEME_DIR/quantize.sh" \
        --model "$MODEL_PATH" \
        --output "$OUTPUT_DIR"
fi

# ---- 3. 启动服务 ----
if [[ "$ACTION" == "all" || "$ACTION" == "serve" ]]; then
    echo ""
    echo -e "${GREEN}>>> [3/4] 启动 1Cat-vLLM 服务${NC}"
    echo -e "${YELLOW}提示: 服务为前台运行, 请另开终端执行 test/perf${NC}"
    bash "$SCHEME_DIR/serve.sh" \
        --model "$OUTPUT_DIR" \
        --port "$PORT" \
        --gpu-util "$GPU_UTIL" \
        --max-len "$MAX_LEN"
fi

# ---- 4. 测试验证 ----
if [[ "$ACTION" == "all" || "$ACTION" == "test" ]]; then
    echo ""
    echo -e "${GREEN}>>> [4/4] 测试服务${NC}"
    echo -e "${CYAN}检查模型列表:${NC}"
    curl -s "http://localhost:${PORT}/v1/models" | python -m json.tool || echo -e "${RED}服务未启动或不可达${NC}"
    echo ""
    echo -e "${CYAN}对话测试:${NC}"
    curl -s "http://localhost:${PORT}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$(basename "$OUTPUT_DIR")\",
            \"messages\": [{\"role\": \"user\", \"content\": \"你好，请介绍一下你自己\"}]
        }" | python -m json.tool || echo -e "${RED}服务未启动或不可达${NC}"
fi

# ---- 领域精度评测 ----
if [[ "$ACTION" == "eval" ]]; then
    echo ""
    echo -e "${GREEN}>>> 领域精度评测 (需服务已启动)${NC}"
    bash "$SCHEME_DIR/benchmark.sh" \
        --base-url "http://localhost:${PORT}" \
        --model "$(basename "$OUTPUT_DIR")" \
        --output "results/domain_$(basename "$OUTPUT_DIR").json"
fi

# ---- 性能测试 ----
if [[ "$ACTION" == "perf" ]]; then
    echo ""
    echo -e "${GREEN}>>> 性能测试 (需服务已启动)${NC}"
    echo -e "${YELLOW}提示: 可复用项目 benchmark_eval.py 的 --perf-test 功能${NC}"
    echo -e "${CYAN}示例:${NC}"
    echo "  source /app/1cat-venv/bin/activate"
    echo "  python llm_deploy/benchmark_eval.py --model $OUTPUT_DIR --perf-test --num-prompts 100 --concurrency 10"
fi

echo ""
echo "=========================================="
echo "完成"
echo "=========================================="
