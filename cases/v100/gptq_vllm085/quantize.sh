#!/usr/bin/env bash
# =============================================================================
# GPTQ 量化脚本 (V100 方案 A, gptqmodel 后端)
#
# 功能:
#   使用项目统一量化脚本 quantize_model.py 的 gptqmodel 后端
#   对 Qwen3-8B 做 GPTQ INT4 量化, 产出标准 GPTQ 格式 (quant_method=gptq)
#
# ⚠️ 重要 (实测验证结论):
#   - V100 必须用 gptqmodel 后端 (configs/gptq_4bit_v100_gptqmodel.yaml),
#     产出标准 GPTQ 格式 (quant_method=gptq), vLLM 走 Exllama kernel
#   - 不要用 llmcompressor 后端 (configs/gptq_4bit_v100.yaml), 产出
#     compressed-tensors 格式, V100 加载报错 (min_capability=80)
#   - 量化环境与部署环境必须隔离 (torch 版本不同)
#
# 用法:
#   bash cases/v100/gptq_vllm085/quantize.sh
#   bash cases/v100/gptq_vllm085/quantize.sh --model /app/local_models/Mind-SLLM-Qwen3-8B \
#       --output /volume/models/Mind-SLLM-Qwen3-8B-GPTQ
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数
MODEL_PATH="/app/local_models/Mind-SLLM-Qwen3-8B"
OUTPUT_DIR="/volume/models/Mind-SLLM-Qwen3-8B-GPTQ"
QUANT_CONFIG="configs/gptq_4bit_v100_gptqmodel.yaml"
QUANT_VENV="/app/venv-quant"
PYTHON_BIN="python3"

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "GPTQ 量化 (V100 方案 A, gptqmodel 后端)"
echo "=========================================="
echo -e "  模型:     ${CYAN}${MODEL_PATH}${NC}"
echo -e "  输出:     ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  配置:     ${CYAN}${QUANT_CONFIG}${NC}"
echo "=========================================="

# ---- 1. 激活量化环境 ----
echo -e "${CYAN}[1/2] 激活量化环境: ${QUANT_VENV}${NC}"
if [ ! -d "$QUANT_VENV" ]; then
    echo -e "${RED}错误: 未找到量化环境 ${QUANT_VENV}${NC}"
    echo -e "${YELLOW}请先运行: bash cases/v100/gptq_vllm085/install_env.sh${NC}"
    exit 1
fi
source "$QUANT_VENV/bin/activate"

# ---- 2. 执行 GPTQ 量化 (gptqmodel 后端) ----
echo -e "${CYAN}[2/2] 执行 GPTQ INT4 量化 (gptqmodel 后端)...${NC}"
echo -e "  ${YELLOW}注意: V100 必须用 gptqmodel 后端, 产出标准 GPTQ 格式${NC}"

# 使用项目统一量化脚本 quantize_model.py 的 gptqmodel 后端
# 该方案已在 V100 实测跑通 (产物: /volume/models/Mind-SLLM-Qwen3-8B-GPTQ)
# 注意: 校准数据路径 (data/calibration/*.jsonl) 为相对路径, 需在项目根目录执行
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

python llm_deploy/quantize_model.py \
    --model "$MODEL_PATH" \
    --method gptq \
    --config "$QUANT_CONFIG" \
    --output "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "GPTQ 量化完成"
echo "=========================================="
echo -e "  量化模型: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo "下一步: 使用 cases/v100/gptq_vllm085/serve.sh 启动 vLLM 0.8.5 推理服务"
