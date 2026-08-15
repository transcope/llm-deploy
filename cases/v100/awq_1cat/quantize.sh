#!/usr/bin/env bash
# =============================================================================
# AWQ 量化脚本 (1Cat-vLLM 方案 B, V100 专用)
#
# 功能:
#   1. 创建独立量化环境 venv-quant-awq (避免与 1Cat-vLLM 的 torch 版本冲突)
#   2. 安装 autoawq (产出 AWQ 原生格式, 1Cat-vLLM SM70 内核所需)
#   3. 使用项目统一量化脚本 quantize_model.py 的 legacy AutoAWQ 路径
#      (quantize_awq_legacy) 对 Qwen3-8B 做非对称 AWQ 量化
#
# ⚠️ 重要 (实测验证结论):
#   - 1Cat-vLLM 的 SM70 内核只支持 **AWQ 原生格式** (quant_method=awq,
#     权重键 qweight/qzeros/scales)
#   - **必须用 AutoAWQ 量化** (quantize_model.py 的 legacy 路径)。
#     不要用 llmcompressor (产出 compressed-tensors 格式, quant_method=
#     compressed-tensors), 1Cat-vLLM 的 --quantization awq 无法加载它,
#     会报 "Quantization method ... does not match" 错误。
#   - 非对称 AWQ (asymmetric, 带 zero-point) 是 SM70 内核要求
#   - 量化环境与推理环境必须隔离 (torch 版本不同)
#   - 校准数据/样本数/序列长度等参数由 configs/awq_4bit_v100.yaml 控制
#
# 用法:
#   bash cases/v100/awq_1cat/quantize.sh
#   bash cases/v100/awq_1cat/quantize.sh --model /app/local_models/Mind-SLLM-Qwen3-8B \
#       --output /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# 默认参数 (与实测跑通产物一致: AutoAWQ 产出 AWQ 原生格式)
MODEL_PATH="/app/local_models/Mind-SLLM-Qwen3-8B"
OUTPUT_DIR="/volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ"
QUANT_CONFIG="configs/awq_4bit_v100.yaml"
QUANT_VENV="/app/venv-quant-awq"
PYTHON_BIN="python3.12"

# 解析参数 (量化细节参数由 yaml 控制, 此处仅保留兼容)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --samples|--max-seq-len|--dataset) echo -e "${YELLOW}警告: 参数 $1 已由 $QUANT_CONFIG 控制, 忽略${NC}"; shift 2 ;;
        *) echo -e "${RED}未知参数: $1${NC}"; exit 1 ;;
    esac
done

echo "=========================================="
echo "AWQ 量化 (1Cat-vLLM 方案 B, V100 专用)"
echo "=========================================="
echo -e "  模型:     ${CYAN}${MODEL_PATH}${NC}"
echo -e "  输出:     ${CYAN}${OUTPUT_DIR}${NC}"
echo -e "  配置:     ${CYAN}${QUANT_CONFIG}${NC} (W4A16_ASYM 非对称)"
echo "=========================================="

# ---- 1. 创建量化虚拟环境 ----
echo -e "${CYAN}[1/3] 创建量化虚拟环境: ${QUANT_VENV}${NC}"
if [ ! -d "$QUANT_VENV" ]; then
    "$PYTHON_BIN" -m venv "$QUANT_VENV"
    echo -e "  ${GREEN}✓${NC} 量化环境已创建"
else
    echo -e "  ${YELLOW}⚠${NC} 量化环境已存在, 跳过创建"
fi
source "$QUANT_VENV/bin/activate"

# ---- 2. 安装 autoawq ----
echo -e "${CYAN}[2/3] 安装 autoawq (产出 AWQ 原生格式)${NC}"
pip install --upgrade pip
pip install autoawq

# ---- 3. 执行 AWQ 量化 (legacy AutoAWQ 路径) ----
echo -e "${CYAN}[3/3] 执行非对称 AWQ 量化 (AutoAWQ, AWQ 原生格式)...${NC}"
echo -e "  ${YELLOW}注意: 1Cat-vLLM SM70 内核只支持 AWQ 原生格式 + 非对称 (带 zero-point)${NC}"

# 使用项目统一量化脚本 quantize_model.py 的 legacy AutoAWQ 路径
# (quantize_awq_legacy, 产出 quant_method=awq 的 AWQ 原生格式)
# 该方案已在 V100 实测跑通 (产物: /volume/models/Mind-SLLM-Qwen3-8B-AWQ-AutoAWQ)
# 注意: 校准数据路径 (data/calibration/*.jsonl) 为相对路径, 需在项目根目录执行
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."

export PYTORCH_ALLOC_CONF=expandable_segments:True
# 强制走 legacy AutoAWQ 路径 (跳过 llmcompressor, 因其产出 compressed-tensors 格式不兼容)
python llm_deploy/quantize_model.py \
    --model "$MODEL_PATH" \
    --method awq \
    --config "$QUANT_CONFIG" \
    --output "$OUTPUT_DIR" \
    --force-legacy-awq

echo ""
echo "=========================================="
echo "AWQ 量化完成"
echo "=========================================="
echo -e "  量化模型: ${GREEN}${OUTPUT_DIR}${NC}"
echo ""
echo "下一步: 使用 cases/v100/awq_1cat/serve.sh 启动 1Cat-vLLM 推理服务"
echo ""
echo "MoE 模型注意: 若是 MoE 架构 (如 Qwen3 MoE), 需在 awq_4bit_v100.yaml 的 ignore 中忽略路由门控层:"
