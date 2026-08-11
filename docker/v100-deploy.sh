#!/bin/bash
# =============================================================================
# V100 一键部署脚本
# 用法: ./v100-deploy.sh <模型名称> [选项]
#
# 示例:
#   ./v100-deploy.sh qwen2.5-7b              # 单卡部署 7B
#   ./v100-deploy.sh qwen2.5-32b             # 多卡部署 32B
#   ./v100-deploy.sh deepseek-r1-14b         # 部署 DeepSeek 蒸馏
#   ./v100-deploy.sh qwen2.5-vl-7b           # 多模态部署
#   ./v100-deploy.sh --list                  # 列出支持的模型
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

# V100 环境变量
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# =============================================================================
# 模型配置数据库 (使用函数返回配置字符串，兼容 bash 3.2+)
# 配置格式: MODEL_ID|TP_SIZE|DTYPE|MAX_LEN|TRUST_CODE|TOOL_PARSER|QUANT
# =============================================================================
get_model_config() {
    local name="$1"
    case "$name" in
        # 注意: V100 (SM 7.0) 不支持 bfloat16, 所有条目必须使用 float16
        qwen2.5-7b)        echo "Qwen/Qwen2.5-7B-Instruct|1|float16|32768|true||" ;;
        qwen2.5-7b-gptq)   echo "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4|1|float16|32768|true||gptq" ;;
        qwen2.5-7b-awq)    echo "Qwen/Qwen2.5-7B-Instruct-AWQ|1|float16|32768|true||awq" ;;
        qwen2.5-14b)       echo "Qwen/Qwen2.5-14B-Instruct|1|float16|32768|true||" ;;
        qwen2.5-14b-gptq)  echo "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4|1|float16|32768|true||gptq" ;;
        qwen2.5-14b-awq)   echo "Qwen/Qwen2.5-14B-Instruct-AWQ|1|float16|32768|true||awq" ;;
        qwen2.5-32b)       echo "Qwen/Qwen2.5-32B-Instruct|2|float16|32768|true||" ;;
        qwen2.5-32b-gptq)  echo "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4|1|float16|32768|true||gptq" ;;
        qwen2.5-32b-awq)   echo "Qwen/Qwen2.5-32B-Instruct-AWQ|1|float16|32768|true||awq" ;;
        qwen2.5-72b)       echo "Qwen/Qwen2.5-72B-Instruct|4|float16|32768|true||" ;;
        qwen2.5-72b-gptq)  echo "Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4|2|float16|32768|true||gptq" ;;
        qwen2.5-72b-awq)   echo "Qwen/Qwen2.5-72B-Instruct-AWQ|2|float16|32768|true||awq" ;;
        qwen3-8b)          echo "Qwen/Qwen3-8B|1|float16|32768|true||" ;;
        qwen3-14b)         echo "Qwen/Qwen3-14B|1|float16|32768|true||" ;;
        qwen3-32b)         echo "Qwen/Qwen3-32B|2|float16|32768|true||" ;;
        qwen3-30b-a3b)     echo "Qwen/Qwen3-30B-A3B|2|float16|32768|true||" ;;
        deepseek-r1-1.5b)  echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|1|float16|8192|true||" ;;
        deepseek-r1-7b)    echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|1|float16|8192|true|deepseek_v3|" ;;
        deepseek-r1-14b)   echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|1|float16|8192|true|deepseek_v3|" ;;
        deepseek-r1-32b)   echo "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|2|float16|8192|true|deepseek_v3|" ;;
        qwen2.5-vl-3b)     echo "Qwen/Qwen2.5-VL-3B-Instruct|1|float16|32768|true||" ;;
        qwen2.5-vl-7b)     echo "Qwen/Qwen2.5-VL-7B-Instruct|1|float16|32768|true||" ;;
        qwen2.5-vl-72b)    echo "Qwen/Qwen2.5-VL-72B-Instruct|4|float16|32768|true||" ;;
        *)                 echo "" ;;
    esac
}

# =============================================================================
# 帮助信息
# =============================================================================
show_help() {
    cat << 'EOF'
用法: ./v100-deploy.sh <模型名称> [选项]

支持的模型:
  ┌─────────────────────────┬─────────────────────────────────────┬──────────┐
  │ 模型名称                │ HuggingFace ID                      │ 所需GPU  │
  ├─────────────────────────┼─────────────────────────────────────┼──────────┤
  │ qwen2.5-7b              │ Qwen/Qwen2.5-7B-Instruct           │ 1x V100  │
  │ qwen2.5-7b-gptq         │ Qwen2.5-7B-Instruct-GPTQ-Int4 ★推荐 │ 1x V100  │
  │ qwen2.5-7b-awq          │ Qwen/Qwen2.5-7B-Instruct-AWQ       │ 1x V100  │
  │ qwen2.5-14b             │ Qwen/Qwen2.5-14B-Instruct          │ 1x V100  │
  │ qwen2.5-14b-gptq        │ Qwen2.5-14B-Instruct-GPTQ-Int4 ★推荐│ 1x V100  │
  │ qwen2.5-14b-awq         │ Qwen/Qwen2.5-14B-Instruct-AWQ      │ 1x V100  │
  │ qwen2.5-32b             │ Qwen/Qwen2.5-32B-Instruct          │ 2x V100  │
  │ qwen2.5-32b-gptq        │ Qwen2.5-32B-Instruct-GPTQ-Int4 ★推荐│ 1x V100  │
  │ qwen2.5-32b-awq         │ Qwen/Qwen2.5-32B-Instruct-AWQ      │ 1x V100  │
  │ qwen2.5-72b             │ Qwen/Qwen2.5-72B-Instruct          │ 4x V100  │
  │ qwen2.5-72b-gptq        │ Qwen2.5-72B-Instruct-GPTQ-Int4 ★推荐│ 2x V100  │
  │ qwen2.5-72b-awq         │ Qwen/Qwen2.5-72B-Instruct-AWQ      │ 2x V100  │
  ├─────────────────────────┼─────────────────────────────────────┼──────────┤
  │ deepseek-r1-1.5b        │ DeepSeek-R1-Distill-Qwen-1.5B      │ 1x V100  │
  │ deepseek-r1-7b          │ DeepSeek-R1-Distill-Qwen-7B        │ 1x V100  │
  │ deepseek-r1-14b         │ DeepSeek-R1-Distill-Qwen-14B       │ 1x V100  │
  │ deepseek-r1-32b         │ DeepSeek-R1-Distill-Qwen-32B       │ 2x V100  │
  ├─────────────────────────┼─────────────────────────────────────┼──────────┤
  │ qwen2.5-vl-3b           │ Qwen2.5-VL-3B-Instruct             │ 1x V100  │
  │ qwen2.5-vl-7b           │ Qwen2.5-VL-7B-Instruct             │ 1x V100  │
  │ qwen2.5-vl-72b          │ Qwen2.5-VL-72B-Instruct            │ 4x V100  │
  └─────────────────────────┴─────────────────────────────────────┴──────────┘

选项:
  -p, --port <port>        服务端口 (默认: 8000)
  -g, --gpu-util <ratio>   GPU 显存利用率 (默认: 0.90)
  -m, --max-len <len>      最大序列长度
  -t, --tp <size>          张量并行大小 (覆盖默认配置)
  --devices <ids>          指定使用的 GPU (如 "2,3", 默认从 GPU 0 开始)
  --awq                    强制使用 AWQ 量化 (如果可用)
  --gptq                   强制使用 GPTQ 量化
  --bnb                    使用 BitsAndBytes NF4 动态量化
  --gptqmodel              用 gptqmodel + TORCH backend 部署 (支持 Qwen3, V100 兼容)
  -d, --dry-run            仅打印命令, 不执行
  --no-prefix-cache        禁用前缀缓存
  -h, --help               显示此帮助

V100 架构限制:
  ❌ 不支持 FP8 量化 (需要 H100+)
  ❌ 不支持 AWQ GEMM kernel (需要 RTX 20+/A100+)
  ❌ 不支持 FlashAttention-2
  ❌ 不支持 bfloat16 (本脚本统一使用 float16)
  ✅ 推荐: GPTQ INT4 / BitsAndBytes NF4 / 原始 FP16

Qwen3 部署注意:
  ⚠️ vLLM 0.7.1 不支持 Qwen3 架构 (报错: Model architectures ['Qwen3ForCausalLM'] are not supported)
  ⚠️ 部署 Qwen3 需用 --gptqmodel 选项 (gptqmodel + TORCH backend, V100 兼容)

示例:
  ./v100-deploy.sh qwen2.5-7b-awq                    # 部署 7B AWQ
  ./v100-deploy.sh deepseek-r1-14b --port 8080       # 部署 14B, 端口8080
  ./v100-deploy.sh qwen2.5-32b --tp 4                # 4卡部署 32B
  ./v100-deploy.sh qwen2.5-7b --bnb --dry-run        # 查看 NF4 部署命令
  ./v100-deploy.sh qwen3-8b --gptqmodel              # 用 gptqmodel 部署 Qwen3 (V100 兼容)

EOF
}

show_list() {
    echo -e "${CYAN}支持的模型列表:${NC}"
    echo ""
    printf "%-25s %-45s %s\n" "模型名称" "HuggingFace ID" "所需GPU"
    echo "───────────────────────── ───────────────────────────────────── ──────────"
    local names=(
        qwen2.5-7b qwen2.5-7b-gptq qwen2.5-7b-awq
        qwen2.5-14b qwen2.5-14b-gptq qwen2.5-14b-awq
        qwen2.5-32b qwen2.5-32b-gptq qwen2.5-32b-awq
        qwen2.5-72b qwen2.5-72b-gptq qwen2.5-72b-awq
        qwen3-8b qwen3-14b qwen3-32b qwen3-30b-a3b
        deepseek-r1-1.5b deepseek-r1-7b deepseek-r1-14b deepseek-r1-32b
        qwen2.5-vl-3b qwen2.5-vl-7b qwen2.5-vl-72b
    )
    for key in "${names[@]}"; do
        local config
        config=$(get_model_config "$key")
        IFS='|' read -r model_id tp _ _ _ _ _ <<< "$config"
        printf "${GREEN}%-25s${NC} %-45s %s\n" "$key" "$model_id" "${tp}x V100"
    done
}

# =============================================================================
# 解析配置
# =============================================================================
parse_config() {
    local model_name="$1"
    local config
    config=$(get_model_config "$model_name")

    if [[ -z "$config" ]]; then
        echo -e "${RED}错误: 不支持的模型 '$model_name'${NC}"
        echo -e "使用 ${YELLOW}--list${NC} 查看支持的模型"
        exit 1
    fi

    IFS='|' read -r MODEL_ID TP_SIZE DTYPE MAX_LEN TRUST_CODE TOOL_PARSER QUANT <<< "$config"
}

# =============================================================================
# 构建 vLLM 命令
# =============================================================================
build_command() {
    local cmd=("vllm" "serve" "$MODEL_ID")

    # 数据类型
    if [[ -n "$DTYPE" ]]; then
        cmd+=("--dtype" "$DTYPE")
    fi

    # 量化
    if [[ -n "$QUANT" ]]; then
        cmd+=("--quantization" "$QUANT")
    fi

    # 张量并行
    if [[ "$TP_SIZE" -gt 1 ]]; then
        cmd+=("--tensor-parallel-size" "$TP_SIZE")
    fi

    # 最大序列长度
    if [[ -n "$MAX_LEN" ]]; then
        cmd+=("--max-model-len" "$MAX_LEN")
    fi

    # 信任远程代码 (Qwen/DeepSeek 必需)
    if [[ "$TRUST_CODE" == "true" ]]; then
        cmd+=("--trust-remote-code")
    fi

    # 工具调用解析器
    if [[ -n "$TOOL_PARSER" ]]; then
        cmd+=("--tool-call-parser" "$TOOL_PARSER")
    fi

    # GPU 显存利用率
    cmd+=("--gpu-memory-utilization" "$GPU_UTIL")

    # 前缀缓存
    if [[ "$PREFIX_CACHE" == "true" ]]; then
        cmd+=("--enable-prefix-caching")
    fi

    # 分块预填充
    cmd+=("--enable-chunked-prefill")

    # 端口
    cmd+=("--port" "$PORT")

    # 主机
    cmd+=("--host" "0.0.0.0")

    echo "${cmd[@]}"
}

# =============================================================================
# 主函数
# =============================================================================
main() {
    # 默认参数
    PORT=8000
    GPU_UTIL=0.90
    PREFIX_CACHE=true
    DRY_RUN=false
    FORCE_QUANT=""
    DEVICES=""
    USE_GPTQMODEL=false

    # 检查参数
    if [[ $# -eq 0 ]]; then
        show_help
        exit 0
    fi

    # 解析参数
    MODEL_NAME=""
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            --list)
                show_list
                exit 0
                ;;
            -p|--port)
                PORT="$2"
                shift 2
                ;;
            -g|--gpu-util)
                GPU_UTIL="$2"
                shift 2
                ;;
            -m|--max-len)
                MAX_LEN="$2"
                shift 2
                ;;
            -t|--tp)
                TP_SIZE="$2"
                shift 2
                ;;
            --devices)
                DEVICES="$2"
                shift 2
                ;;
            --awq)
                FORCE_QUANT="awq"
                shift
                ;;
            --gptq)
                FORCE_QUANT="gptq"
                shift
                ;;
            --bnb)
                FORCE_QUANT="bitsandbytes"
                shift
                ;;
            --gptqmodel)
                USE_GPTQMODEL=true
                shift
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --no-prefix-cache)
                PREFIX_CACHE=false
                shift
                ;;
            -*)
                echo -e "${RED}错误: 未知选项 $1${NC}"
                show_help
                exit 1
                ;;
            *)
                if [[ -z "$MODEL_NAME" ]]; then
                    MODEL_NAME="$1"
                fi
                shift
                ;;
        esac
    done

    if [[ -z "$MODEL_NAME" ]]; then
        echo -e "${RED}错误: 请指定模型名称${NC}"
        show_help
        exit 1
    fi

    # 解析模型配置
    parse_config "$MODEL_NAME"

    # 应用强制量化
    if [[ -n "$FORCE_QUANT" ]]; then
        QUANT="$FORCE_QUANT"
        # BitsAndBytes 需要特殊 dtype
        if [[ "$FORCE_QUANT" == "bitsandbytes" ]]; then
            DTYPE="float16"
        fi
    fi

    # V100 警告检查
    if [[ "$QUANT" == "awq" ]]; then
        echo -e "${YELLOW}⚠️  警告: AWQ 在 V100 (SM 7.0) 上可能使用 GEMV kernel (较慢)${NC}"
        echo -e "${YELLOW}   建议改用 GPTQ 量化模型以获得更好性能${NC}"
        echo ""
    fi

    if [[ "$QUANT" == "fp8" ]]; then
        echo -e "${RED}❌ 错误: V100 不支持 FP8 量化!${NC}"
        echo -e "${YELLOW}   V100 支持的量化: GPTQ INT4, BitsAndBytes NF4, SmoothQuant W8A8${NC}"
        exit 1
    fi

    # 计算所需 GPU
    REQUIRED_GPUS="${TP_SIZE:-1}"

    # 显示部署信息
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════════╗"
    echo "║                    V100 模型部署配置                              ║"
    echo "╚══════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo -e "  模型:       ${GREEN}$MODEL_ID${NC}"
    echo -e "  量化:       ${GREEN}${QUANT:-无 (FP16)}${NC}"
    echo -e "  数据类型:   ${GREEN}${DTYPE:-auto}${NC}"
    echo -e "  张量并行:   ${GREEN}${REQUIRED_GPUS} GPU(s)${NC}"
    echo -e "  最大长度:   ${GREEN}${MAX_LEN:-模型默认}${NC}"
    echo -e "  显存利用率: ${GREEN}${GPU_UTIL}${NC}"
    echo -e "  前缀缓存:   ${GREEN}${PREFIX_CACHE}${NC}"
    echo -e "  服务端口:   ${GREEN}${PORT}${NC}"
    echo ""

    # 构建命令
    CMD=$(build_command)

    echo -e "${CYAN}启动命令:${NC}"
    echo -e "  ${YELLOW}$CMD${NC}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        echo -e "${YELLOW}[干运行模式] 不实际启动服务${NC}"
        exit 0
    fi

    # 检查 GPU 可用性
    AVAILABLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
    if [[ "$AVAILABLE_GPUS" -lt "$REQUIRED_GPUS" ]]; then
        echo -e "${RED}错误: 需要 ${REQUIRED_GPUS} 块 GPU, 但只检测到 ${AVAILABLE_GPUS} 块${NC}"
        exit 1
    fi

    # 设置 CUDA_VISIBLE_DEVICES: 优先使用 --devices 指定的 GPU 列表
    if [[ -n "$DEVICES" ]]; then
        DEVICE_COUNT=$(echo "$DEVICES" | tr ',' '\n' | grep -c .)
        if [[ "$DEVICE_COUNT" -ne "$REQUIRED_GPUS" ]]; then
            echo -e "${RED}错误: --devices 指定了 ${DEVICE_COUNT} 块 GPU, 但张量并行需要 ${REQUIRED_GPUS} 块${NC}"
            exit 1
        fi
        export CUDA_VISIBLE_DEVICES="$DEVICES"
    else
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((REQUIRED_GPUS - 1)))
    fi
    echo -e "${GREEN}使用 GPU: $CUDA_VISIBLE_DEVICES${NC}"
    echo ""

    # 启动服务
    if [[ "$USE_GPTQMODEL" == "true" ]]; then
        # gptqmodel + TORCH backend 部署 (支持 Qwen3, V100 兼容)
        # 注意: vLLM 0.7.1 不支持 Qwen3 架构, 需用 gptqmodel 部署
        echo -e "${GREEN}正在启动 gptqmodel 服务 (TORCH backend, 支持 Qwen3)...${NC}"
        echo -e "${CYAN}服务地址: http://0.0.0.0:${PORT}${NC}"
        echo ""
        echo -e "${YELLOW}使用 serve_gptq.py 部署 (OpenAI 兼容 API)${NC}"
        echo -e "${YELLOW}  python serve_gptq.py --model $MODEL_ID --host 0.0.0.0 --port $PORT${NC}"
        echo ""
        # 检查 serve_gptq.py 是否存在
        if [[ -f "/volume/workspace/llm-deploy/serve_gptq.py" ]]; then
            exec /app/venv-deploy/bin/python /volume/workspace/llm-deploy/serve_gptq.py \
                --model "$MODEL_ID" --host 0.0.0.0 --port "$PORT"
        elif [[ -f "serve_gptq.py" ]]; then
            exec /app/venv-deploy/bin/python serve_gptq.py \
                --model "$MODEL_ID" --host 0.0.0.0 --port "$PORT"
        else
            echo -e "${RED}错误: 未找到 serve_gptq.py, 请先将其复制到项目目录${NC}"
            exit 1
        fi
    else
        # vLLM 部署
        echo -e "${GREEN}正在启动 vLLM 服务...${NC}"
        echo -e "${CYAN}服务地址: http://0.0.0.0:${PORT}${NC}"
        echo -e "${CYAN}API 文档: http://0.0.0.0:${PORT}/docs${NC}"
        echo ""
        eval "$CMD"
    fi
}

main "$@"
