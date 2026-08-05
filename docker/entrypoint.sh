#!/bin/bash
# =============================================================================
# Docker 入口脚本 - V100 大模型推理环境
# =============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║          V100 大模型推理部署环境                                  ║"
echo "║          LLM Inference on NVIDIA V100 (Volta, SM 7.0)            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# 检查 GPU 环境
# =============================================================================
echo -e "${YELLOW}[1/5] 检查 GPU 环境...${NC}"

if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: nvidia-smi 未找到, 请确保 NVIDIA Docker Runtime 已正确配置${NC}"
    exit 1
fi

echo -e "${GREEN}GPU 信息:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
    --format=csv,noheader 2>/dev/null || nvidia-smi -L

GPU_COUNT=$(nvidia-smi -L | wc -l)
echo -e "${GREEN}检测到 ${GPU_COUNT} 块 GPU${NC}"

# 检查 CUDA 可用性
echo -e "${YELLOW}[2/5] 检查 PyTorch CUDA 支持...${NC}"
python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 版本: {torch.version.cuda}')
print(f'cuDNN 版本: {torch.backends.cudnn.version()}')
print(f'GPU 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)')
        # 检查计算能力
        major = props.major
        minor = props.minor
        print(f'    计算能力: {major}.{minor}')
        if major < 7:
            print(f'    ⚠️  警告: 计算能力 {major}.{minor} 可能不受完全支持')
        elif major == 7:
            print(f'    ✅ Volta 架构 (V100) - 注意: 不支持 FP8/AWQ GEMM/FlashAttn-2')
        elif major == 8:
            print(f'    ✅ Ampere 架构 (A100/RTX 30xx)')
        elif major == 9:
            print(f'    ✅ Hopper 架构 (H100) - 支持 FP8')
"

# =============================================================================
# 检查 vLLM
# =============================================================================
echo -e "${YELLOW}[3/5] 检查 vLLM 安装...${NC}"
python -c "
import vllm
print(f'vLLM 版本: {vllm.__version__}')
print(f'vLLM V1 引擎: {getattr(vllm, \"V1\", False) or \"默认\"}')
"

# =============================================================================
# 检查量化工具
# =============================================================================
echo -e "${YELLOW}[4/5] 检查量化工具链...${NC}"

# GPTQModel
python -c "import gptqmodel; print('✅ GPTQModel: 已安装')" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  GPTQModel: 未安装 (可选)${NC}"

# BitsAndBytes
python -c "import bitsandbytes; print('✅ BitsAndBytes: 已安装')" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  BitsAndBytes: 未安装${NC}"

# llm-compressor
python -c "import llmcompressor; print('✅ llm-compressor: 已安装')" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  llm-compressor: 未安装${NC}"

# lm-eval
python -c "import lm_eval; print('✅ lm-eval: 已安装')" 2>/dev/null || \
    echo -e "${YELLOW}⚠️  lm-eval: 未安装${NC}"

# =============================================================================
# 检查磁盘空间
# =============================================================================
echo -e "${YELLOW}[5/5] 检查磁盘空间...${NC}"
df -h /app | awk 'NR==2 {printf "可用空间: %s / %s (已用 %s)\n", $4, $2, $5}'

# 创建必要目录
mkdir -p /app/models /app/results /app/cache/huggingface

# =============================================================================
# V100 特定警告
# =============================================================================
echo ""
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}  V100 架构限制提醒:${NC}"
echo -e "${YELLOW}    ❌ 不支持 FP8 量化 (需要 Hopper SM 90+)${NC}"
echo -e "${YELLOW}    ❌ 不支持 AWQ GEMM kernel (需要 Turing SM 75+)${NC}"
echo -e "${YELLOW}    ❌ 不支持 FlashAttention-2 (需要 Ampere SM 80+)${NC}"
echo -e "${YELLOW}    ✅ 支持 GPTQ INT4 (EXL2 kernel, SM 70+) ★推荐${NC}"
echo -e "${YELLOW}    ✅ 支持 BitsAndBytes NF4 (动态量化)${NC}"
echo -e "${YELLOW}    ✅ 支持 SmoothQuant W8A8 (INT8)${NC}"
echo -e "${YELLOW}════════════════════════════════════════════════════════════════${NC}"

echo ""
echo -e "${GREEN}环境准备完成!${NC}"
echo -e "${BLUE}可用命令:${NC}"
echo -e "  ${GREEN}./v100-deploy.sh <model> [options]${NC}    - 快速部署模型"
echo -e "  ${GREEN}python src/quantize_model.py${NC}  - 量化模型"
echo -e "  ${GREEN}python src/benchmark_eval.py${NC}  - 评测模型"
echo -e "  ${GREEN}vllm serve <model> [options]${NC}     - 直接启动 vLLM"
echo ""

# 执行传入的命令
exec "$@"
