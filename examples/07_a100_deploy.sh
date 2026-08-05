#!/usr/bin/env bash
# =============================================================================
# A100 单卡端到端部署脚本 (量化 → 部署 → 评测)
#
# A100 (SM 8.0, Ampere) 优势:
#   ✅ 原生支持 bfloat16 (比 float16 数值更稳)
#   ✅ AWQ GEMM kernel 全速运行 (A100 首选量化方案)
#   ✅ Marlin INT4 kernel / FlashAttention-2 可用
#   ❌ 不支持 FP8 (需 H100+)
#
# 用法:
#   ./07_a100_deploy.sh quantize [模型ID] [输出路径]   # 阶段1: AWQ 量化
#   ./07_a100_deploy.sh deploy   [量化模型路径]        # 阶段2: 启动推理服务
#   ./07_a100_deploy.sh eval     [量化模型路径] [基线]  # 阶段3: 精度评测
#   ./07_a100_deploy.sh perf     [服务地址]            # 阶段4: 性能基准
#   ./07_a100_deploy.sh all      [模型ID]              # 一键全流程
#   ./07_a100_deploy.sh --help                         # 帮助
#
# 示例:
#   ./07_a100_deploy.sh quantize                                 # 量化 7B (默认)
#   ./07_a100_deploy.sh quantize Qwen/Qwen2.5-14B-Instruct       # 量化 14B
#   ./07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ           # 部署
#   ./07_a100_deploy.sh eval ./models/Qwen2.5-7B-AWQ             # 评测
#   ./07_a100_deploy.sh perf                                     # 性能测试
#   ./07_a100_deploy.sh all                                      # 全流程 7B
# =============================================================================

set -e

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# A100 推荐默认值
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_OUTPUT="./models/Qwen2.5-7B-AWQ"
DEFAULT_BASELINE="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_BASE_URL="http://localhost:8000"
DEFAULT_PORT=8000
DEFAULT_GPU_UTIL=0.90
DEFAULT_MAX_LEN=32768
DEFAULT_DTYPE="bfloat16"   # A100 原生支持 bfloat16
DEFAULT_QUANT="awq"        # A100 首选 AWQ GEMM
DEFAULT_TASKS="gsm8k,hellaswag"

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    cat << 'EOF'
A100 单卡端到端部署脚本 (量化 → 部署 → 评测)

用法: ./07_a100_deploy.sh <子命令> [参数]

子命令:
  quantize [模型] [输出]    AWQ INT4 量化 (A100 首选)
  deploy   [模型路径]       启动 vLLM 推理服务
  eval     [模型路径] [基线] 精度评测 (lm-eval, 对比基线)
  perf     [服务地址]       性能基准 (吞吐/延迟/TTFT, 需服务已启动)
  all      [模型]           一键全流程: 量化 → 部署 → 评测
  --help                    显示此帮助

默认值:
  模型:      Qwen/Qwen2.5-7B-Instruct
  量化:      awq (INT4, A100 GEMM kernel)
  数据类型:  bfloat16 (A100 原生支持)
  输出路径:  ./models/Qwen2.5-7B-AWQ
  服务端口:  8000
  评测任务:  gsm8k,hellaswag

A100 量化方案选择:
  ★ AWQ INT4    首选 (GEMM kernel, 75% 显存节省, ~95% 精度)
    GPTQ INT4   次选 (通用兼容, EXL2/Marlin kernel)
    W8A8        精度敏感场景 (50% 显存, ~96% 精度)
    原始 BF16   不量化 (最高精度, 显存占用最大)
  ✗ FP8         不支持 (需 H100/H200)

A100 显存参考 (单卡):
  7B  AWQ INT4  ~5GB   → 40GB/80GB 均可
  14B AWQ INT4  ~9GB   → 40GB/80GB 均可
  32B AWQ INT4  ~20GB  → 需 80GB (40GB 需双卡张量并行)

示例:
  # 阶段化执行 (推荐, 可逐步验证)
  ./07_a100_deploy.sh quantize
  ./07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ
  # 另开终端:
  ./07_a100_deploy.sh eval ./models/Qwen2.5-7B-AWQ
  ./07_a100_deploy.sh perf

  # 量化更大模型
  ./07_a100_deploy.sh quantize Qwen/Qwen2.5-14B-Instruct ./models/Qwen2.5-14B-AWQ

  # 一键全流程 (量化+部署+评测, 需保持终端)
  ./07_a100_deploy.sh all

EOF
}

# =============================================================================
# 环境准备
# =============================================================================
activate_env() {
    if [[ -f "${PROJECT_DIR}/vllm-env/bin/activate" ]]; then
        # shellcheck disable=SC1091
        source "${PROJECT_DIR}/vllm-env/bin/activate"
    elif [[ -f "${PROJECT_DIR}/vllm-env/Scripts/activate" ]]; then
        # Windows Git Bash
        # shellcheck disable=SC1091
        source "${PROJECT_DIR}/vllm-env/Scripts/activate"
    else
        echo -e "${YELLOW}[警告] 未找到 vllm-env 虚拟环境, 使用当前 Python 环境${NC}"
    fi
    cd "${PROJECT_DIR}"
}

# =============================================================================
# 阶段 1: AWQ 量化
# =============================================================================
do_quantize() {
    local model="${1:-$DEFAULT_MODEL}"
    local output="${2:-$DEFAULT_OUTPUT}"

    activate_env

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  阶段 1/3: AWQ INT4 量化 (A100 推荐)                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  原始模型:   ${GREEN}${model}${NC}"
    echo -e "  量化方法:   ${GREEN}AWQ INT4 (W4A16, GEMM kernel)${NC}"
    echo -e "  输出路径:   ${GREEN}${output}${NC}"
    echo -e "  配置文件:   ${GREEN}configs/awq_4bit.yaml${NC}"
    echo ""

    if [[ -d "${output}" ]] && [[ -n "$(ls -A "${output}" 2>/dev/null)" ]]; then
        echo -e "${YELLOW}[提示] 输出目录已存在且非空: ${output}${NC}"
        echo -e "${YELLOW}       如需重新量化, 请先删除该目录${NC}"
        echo ""
    fi

    echo -e "${CYAN}[启动量化] 预计耗时 10-40 分钟 (视模型大小而定)...${NC}"
    echo ""

    python scripts/quantize_model.py \
        --model "${model}" \
        --method awq \
        --config configs/awq_4bit.yaml \
        --output "${output}"

    echo ""
    echo -e "${GREEN}✓ 量化完成! 模型保存在: ${output}${NC}"
    echo ""
    echo -e "${CYAN}下一步 - 部署:${NC}"
    echo -e "  ./07_a100_deploy.sh deploy ${output}"
    echo -e "${CYAN}下一步 - 评测:${NC}"
    echo -e "  ./07_a100_deploy.sh eval ${output}"
}

# =============================================================================
# 阶段 2: 启动推理服务
# =============================================================================
do_deploy() {
    local model="${1:-$DEFAULT_OUTPUT}"
    local port="${PORT:-$DEFAULT_PORT}"
    local gpu_util="${GPU_UTIL:-$DEFAULT_GPU_UTIL}"
    local max_len="${MAX_LEN:-$DEFAULT_MAX_LEN}"

    activate_env

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  阶段 2/3: 启动 vLLM 推理服务 (A100 单卡)                     ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  模型路径:   ${GREEN}${model}${NC}"
    echo -e "  量化:       ${GREEN}${DEFAULT_QUANT} (自动从 config.json 识别)${NC}"
    echo -e "  数据类型:   ${GREEN}${DEFAULT_DTYPE} (A100 原生 BF16)${NC}"
    echo -e "  显存利用:   ${GREEN}${gpu_util}${NC}"
    echo -e "  最大长度:   ${GREEN}${max_len}${NC}"
    echo -e "  服务端口:   ${GREEN}${port}${NC}"
    echo ""

    if [[ ! -d "${model}" ]]; then
        echo -e "${YELLOW}[提示] 本地未找到模型目录: ${model}${NC}"
        echo -e "${YELLOW}       将作为 HuggingFace 模型 ID 在线加载${NC}"
        echo ""
    fi

    echo -e "${CYAN}[启动服务] 启动后访问 http://localhost:${port}${NC}"
    echo -e "${CYAN}          按 Ctrl+C 停止服务${NC}"
    echo ""
    echo -e "${YELLOW}提示: 评测/性能测试请在另一个终端执行 (服务需保持运行)${NC}"
    echo ""

    # deploy_server.py 会自动从 config.json 识别量化方式, 并按 GPU 能力校验参数
    # A100 (SM 8.0) 不会被强制降级, bfloat16 / AWQ 全部按传入值生效
    python scripts/deploy_server.py \
        --model "${model}" \
        --dtype "${DEFAULT_DTYPE}" \
        --gpu-util "${gpu_util}" \
        --max-model-len "${max_len}" \
        --enable-prefix-caching \
        --enable-chunked-prefill \
        --trust-remote-code \
        --port "${port}"
}

# =============================================================================
# 阶段 3a: 精度评测
# =============================================================================
do_eval() {
    local model="${1:-$DEFAULT_OUTPUT}"
    local baseline="${2:-$DEFAULT_BASELINE}"

    activate_env

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  阶段 3/3: 精度评测 (lm-eval, 对比基线模型)                   ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  量化模型:   ${GREEN}${model}${NC}"
    echo -e "  基线模型:   ${GREEN}${baseline}${NC}"
    echo -e "  评测任务:   ${GREEN}${DEFAULT_TASKS}${NC}"
    echo -e "  数据类型:   ${GREEN}${DEFAULT_DTYPE}${NC}"
    echo ""

    echo -e "${CYAN}[启动评测] 将分别评测基线与量化模型, 对比精度损失...${NC}"
    echo -e "${CYAN}          预计耗时 20-60 分钟 (视任务与模型而定)${NC}"
    echo ""

    # A100 资源充足, 使用默认 gpu_memory_utilization=0.8 即可
    # dtype=bfloat16: A100 原生支持, 比 float16 数值更稳
    python scripts/benchmark_eval.py \
        --model "${model}" \
        --quantization "${DEFAULT_QUANT}" \
        --dtype "${DEFAULT_DTYPE}" \
        --tasks "${DEFAULT_TASKS}" \
        --baseline-model "${baseline}" \
        --output "./results/a100_awq_comparison"

    echo ""
    echo -e "${GREEN}✓ 评测完成! 结果保存在: ./results/a100_awq_comparison${NC}"
    echo ""
    echo -e "${CYAN}性能测试 (需服务已启动):${NC}"
    echo -e "  ./07_a100_deploy.sh perf"
}

# =============================================================================
# 阶段 3b: 性能基准测试
# =============================================================================
do_perf() {
    local base_url="${1:-$DEFAULT_BASE_URL}"

    activate_env

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  性能基准测试 (吞吐 / 延迟 / TTFT)                            ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  服务地址:   ${GREEN}${base_url}${NC}"
    echo -e "  请求数:     ${GREEN}100${NC}"
    echo -e "  并发数:     ${GREEN}10${NC}"
    echo -e "  最大token:  ${GREEN}256${NC}"
    echo ""

    # 先检查服务是否在线
    if ! curl -s --max-time 5 "${base_url}/v1/models" >/dev/null 2>&1; then
        echo -e "${RED}错误: 服务未启动或无法访问: ${base_url}${NC}"
        echo -e "${YELLOW}       请先在另一终端执行: ./07_a100_deploy.sh deploy${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ 服务在线${NC}"
    echo ""
    echo -e "${CYAN}[启动性能测试]...${NC}"
    echo ""

    # --skip-accuracy: 跳过精度评测, 直接测性能 (避免与运行中的服务争抢显存)
    python scripts/benchmark_eval.py \
        --model "${DEFAULT_OUTPUT}" \
        --perf-test \
        --skip-accuracy \
        --base-url "${base_url}" \
        --num-prompts 100 \
        --max-tokens 256 \
        --concurrency 10 \
        --output "./results/a100_perf"

    echo ""
    echo -e "${GREEN}✓ 性能测试完成! 结果保存在: ./results/a100_perf${NC}"
}

# =============================================================================
# 一键全流程
# =============================================================================
do_all() {
    local model="${1:-$DEFAULT_MODEL}"
    local output="${2:-$DEFAULT_OUTPUT}"

    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║  A100 一键全流程: 量化 → 部署 → 评测                          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  原始模型: ${GREEN}${model}${NC}"
    echo -e "  输出路径: ${GREEN}${output}${NC}"
    echo ""

    # 阶段 1: 量化
    do_quantize "${model}" "${output}"

    # 阶段 2: 后台启动服务 (评测需要服务在线? 精度评测走 lm-eval vllm 直连, 不需要)
    # 精度评测 (lm-eval) 会自己启动 vLLM 实例, 不依赖外部服务
    # 性能测试才需要外部服务, 全流程里放在评测之后单独提示

    # 阶段 3: 精度评测
    do_eval "${output}" "${model}"

    echo ""
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  全流程完成!${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  量化模型:   ${CYAN}${output}${NC}"
    echo -e "  精度结果:   ${CYAN}./results/a100_awq_comparison${NC}"
    echo ""
    echo -e "${YELLOW}下一步 - 启动服务并测试性能:${NC}"
    echo -e "  ./07_a100_deploy.sh deploy ${output}"
    echo -e "  # 另开终端:"
    echo -e "  ./07_a100_deploy.sh perf"
}

# =============================================================================
# 主入口
# =============================================================================
main() {
    if [[ $# -eq 0 ]] || [[ "$1" == "-h" ]] || [[ "$1" == "--help" ]]; then
        show_help
        exit 0
    fi

    local subcommand="$1"
    shift

    case "${subcommand}" in
        quantize)
            do_quantize "$@"
            ;;
        deploy)
            # 支持通过环境变量覆盖端口/显存/长度: PORT=9000 ./07_a100_deploy.sh deploy ...
            do_deploy "$@"
            ;;
        eval)
            do_eval "$@"
            ;;
        perf)
            do_perf "$@"
            ;;
        all)
            do_all "$@"
            ;;
        *)
            echo -e "${RED}错误: 未知子命令 '${subcommand}'${NC}"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
