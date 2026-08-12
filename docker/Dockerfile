# =============================================================================
# V100 大模型推理部署 Docker 镜像
# 适配: NVIDIA V100 (Volta, SM 7.0), CUDA 12.6 / Ubuntu 24.04
# 注意: V100 不支持 FP8、AWQ GEMM、FlashAttention-2
#
# 构建说明:
#   - 宿主机 NVIDIA 驱动只需 >= 12.4 即可运行 CUDA 12.6 容器
#   - 使用国内 apt / pip 镜像加速构建
#   - 如不需要国内源, 注释掉对应的 RUN 行即可
# =============================================================================

FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

# 构建参数: 是否只安装基础环境 (torch + 系统依赖 + pip 配置)
# - true:  跳过 vLLM、量化工具链、HuggingFace/评测/数据科学等所有后续 Python 包安装
# - false: 按下面各参数分别控制 (默认)
ARG INSTALL_BASE_ONLY=false

# 构建参数: 是否自动安装 vLLM 推理引擎
# - true:  构建时自动安装 (默认)
# - false: 构建时跳过, 进入容器后手动安装 vllm
ARG INSTALL_VLLM=true

# 构建参数: 是否自动安装量化工具链
# - true:  构建时自动安装 (默认)
# - false: 构建时跳过, 可在运行后手动执行 bash /app/cases/v100/install_quant_tools.sh
ARG INSTALL_QUANT_TOOLS=true

LABEL maintainer="LLM Deploy Team"
LABEL description="V100-optimized LLM inference environment with vLLM + GPTQ + BitsAndBytes (Ubuntu 24.04 / CUDA 12.6)"

# =============================================================================
# 基础环境配置
# =============================================================================
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# V100 必需环境变量
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV CUDA_DEVICE_ORDER=PCI_BUS_ID
ENV NCCL_P2P_DISABLE=0
ENV NCCL_IB_DISABLE=0

# 时区
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# =============================================================================
# apt 源: 多个国内镜像 + 官方源保底 (Ubuntu 24.04 noble)
# apt 会按顺序尝试每个源, 任一国内源可用时即可加速, 全部失败时回退官方源
# =============================================================================
RUN rm -f /etc/apt/sources.list.d/*.sources && \
    cat > /etc/apt/sources.list <<'EOF'
# 清华大学
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu noble main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu noble-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu noble-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu noble-security main restricted universe multiverse

# 阿里云
deb https://mirrors.aliyun.com/ubuntu noble main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu noble-updates main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu noble-backports main restricted universe multiverse
deb https://mirrors.aliyun.com/ubuntu noble-security main restricted universe multiverse

# 中国科学技术大学
deb https://mirrors.ustc.edu.cn/ubuntu noble main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu noble-updates main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu noble-backports main restricted universe multiverse
deb https://mirrors.ustc.edu.cn/ubuntu noble-security main restricted universe multiverse

# 腾讯云
deb https://mirrors.cloud.tencent.com/ubuntu noble main restricted universe multiverse
deb https://mirrors.cloud.tencent.com/ubuntu noble-updates main restricted universe multiverse
deb https://mirrors.cloud.tencent.com/ubuntu noble-backports main restricted universe multiverse
deb https://mirrors.cloud.tencent.com/ubuntu noble-security main restricted universe multiverse

# 官方源保底
deb http://archive.ubuntu.com/ubuntu noble main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu noble-updates main restricted universe multiverse
deb http://archive.ubuntu.com/ubuntu noble-backports main restricted universe multiverse
deb http://security.ubuntu.com/ubuntu noble-security main restricted universe multiverse
EOF

# =============================================================================
# 系统依赖安装
# =============================================================================
RUN apt-get update -o Acquire::Retries=3 && apt-get install -y --no-install-recommends -o Acquire::Retries=3 \
    # 基础工具
    build-essential \
    ca-certificates \
    curl \
    wget \
    git \
    git-lfs \
    ssh \
    # Python (Ubuntu 24.04 默认 Python 3.12)
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    # 编译工具
    cmake \
    ninja-build \
    # 数学库
    libopenblas-dev \
    libomp-dev \
    # 实用工具
    vim \
    nano \
    htop \
    tmux \
    jq \
    # 监控
    pciutils \
    nvtop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建 Python 虚拟环境（避免 Ubuntu 24.04 系统 Python 的 PEP 668 限制）
RUN /usr/bin/python3 -m venv /app/venv
ENV VIRTUAL_ENV=/app/venv
ENV PATH="/app/venv/bin:$PATH"

# 升级 pip 并配置大量国内 PyPI 镜像 + 官方源保底
# - 主源: 清华 PyPI
# - 备用: 阿里云 / 中科大 / 腾讯 / 豆瓣 PyPI
# - 官方 PyPI + PyTorch CUDA 12.4 wheel 索引保底
# 解决清华 pytorch-wheels 缺少 torch==2.5.1 的问题
#
# 注意: vllm / torch 等包体积较大(数百 MB 到数 GB), 网络波动时容易超时,
#       因此 timeout 设为 300s, retries 设为 5, 并通过环境变量兜底。
ENV PIP_DEFAULT_TIMEOUT=300
RUN python -m pip install --upgrade pip setuptools wheel && \
    pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.extra-index-url "https://mirrors.aliyun.com/pypi/simple https://pypi.mirrors.ustc.edu.cn/simple https://mirrors.cloud.tencent.com/pypi/simple https://pypi.doubanio.com/simple https://pypi.org/simple" && \
    pip config set global.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com pypi.mirrors.ustc.edu.cn mirrors.cloud.tencent.com pypi.doubanio.com pypi.org download.pytorch.org" && \
    pip config set global.timeout 300 && \
    pip config set global.retries 5

# =============================================================================
# PyTorch (CUDA 12.4 版本, 与 CUDA 12.6 运行时兼容)
# 仅在这一步加入 PyTorch 官方 cu124 wheel 索引, 避免其他包查询该索引浪费时间
# =============================================================================
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1

# =============================================================================
# vLLM 推理引擎 (V100 兼容版本)
# =============================================================================
# V100 (SM 7.0) 注意事项:
# - 不支持 FlashAttention-2, vLLM 会自动 fallback 到 xFormers
# - 需要设置 VLLM_WORKER_MULTIPROC_METHOD=spawn
# - 不支持 FP8 KV Cache
#
# 版本锁定说明:
# - vllm==0.8.x 要求 torch==2.6.0, 与项目锁定的 torch==2.5.1 冲突
# - vllm==0.7.1 要求 torch==2.5.1 / xformers==0.0.28.post3 / compressed-tensors==0.9.0
# - 该组合与 llmcompressor==0.4.0 的 compressed-tensors 要求一致
RUN if [ "$INSTALL_BASE_ONLY" = "true" ]; then \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过 vLLM 自动安装"; \
    elif [ "$INSTALL_VLLM" = "true" ]; then \
        pip install --no-cache-dir \
            --extra-index-url https://download.pytorch.org/whl/cu124 \
            vllm==0.7.1 \
            xformers==0.0.28.post3 \
            transformers>=4.51.0 \
            "numpy<2.0.0"; \
    else \
        echo "[构建参数] INSTALL_VLLM=false, 跳过 vLLM 自动安装"; \
    fi

# =============================================================================
# 量化工具链 (V100 适配版)
# =============================================================================
# V100 支持的量化方案:
# - GPTQ INT4 (EXL2 kernel, SM 70+ 支持) ★推荐
# - BitsAndBytes NF4 (动态量化, 兼容性最好)
# - SmoothQuant W8A8 (INT8, V100 完全支持)
# V100 不支持的量化方案:
# - AWQ GEMM (需要 SM 75+, Turing+)
# - FP8 (需要 SM 90+, Hopper)
# - Marlin kernel (需要 SM 80+, Ampere)

# 量化工具链 (可选, 构建时通过 INSTALL_QUANT_TOOLS 控制)
# 脚本已内置兼容版本, 也支持在容器内手动运行:
#   bash /app/cases/v100/install_quant_tools.sh
COPY cases/v100/install_quant_tools.sh /app/cases/v100/install_quant_tools.sh
RUN chmod +x /app/cases/v100/install_quant_tools.sh && \
    if [ "$INSTALL_BASE_ONLY" = "true" ]; then \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过量化工具链自动安装"; \
    elif [ "$INSTALL_QUANT_TOOLS" = "true" ]; then \
        /app/cases/v100/install_quant_tools.sh; \
    else \
        echo "[构建参数] INSTALL_QUANT_TOOLS=false, 跳过量化工具链自动安装"; \
    fi

# =============================================================================
# HuggingFace 生态
# =============================================================================
RUN if [ "$INSTALL_BASE_ONLY" != "true" ]; then \
        pip install --no-cache-dir \
            transformers>=4.51.0 \
            accelerate>=0.34.0 \
            huggingface-hub>=0.25.0 \
            safetensors>=0.4.0 \
            tokenizers>=0.20.0; \
    else \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过 HuggingFace 生态安装"; \
    fi

# =============================================================================
# 评测工具
# =============================================================================
RUN if [ "$INSTALL_BASE_ONLY" != "true" ]; then \
        pip install --no-cache-dir \
            lm-eval>=0.4.5 \
            datasets>=3.0.0 \
            openai>=1.50.0; \
    else \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过评测工具安装"; \
    fi

# =============================================================================
# 多模态支持
# =============================================================================
RUN if [ "$INSTALL_BASE_ONLY" != "true" ]; then \
        pip install --no-cache-dir \
            Pillow>=10.0.0 \
            requests>=2.32.0 \
            aiohttp>=3.10.0; \
    else \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过多模态支持安装"; \
    fi

# =============================================================================
# 数据科学 & 可视化
# =============================================================================
RUN if [ "$INSTALL_BASE_ONLY" != "true" ]; then \
        pip install --no-cache-dir \
            "numpy>=1.26.0,<2.0.0" \
            scipy>=1.14.0 \
            pandas>=2.2.0 \
            matplotlib>=3.9.0 \
            tqdm>=4.66.0; \
    else \
        echo "[构建参数] INSTALL_BASE_ONLY=true, 跳过数据科学工具安装"; \
    fi

# =============================================================================
# 安装项目脚本
# =============================================================================
WORKDIR /app

# 复制项目文件
COPY llm_deploy/ /app/llm_deploy/
COPY configs/ /app/configs/
COPY requirements-quant.txt requirements-vllm085.txt /app/
COPY docker/entrypoint.sh /app/
COPY docker/v100-deploy.sh /app/

RUN chmod +x /app/entrypoint.sh /app/v100-deploy.sh /app/llm_deploy/*.py

# =============================================================================
# 验证安装
# =============================================================================
RUN if [ "$INSTALL_BASE_ONLY" != "true" ]; then \
        python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')" \
        && (python -c "import vllm; print(f'vLLM: {vllm.__version__}')" 2>/dev/null || echo "vLLM: optional/skipped") \
        && python -c "import transformers; print(f'Transformers: {transformers.__version__}')" \
        && (python -c "import compressed_tensors; print(f'compressed-tensors: OK')" 2>/dev/null || echo "compressed-tensors: optional/skipped") \
        && (python -c "from llmcompressor.modifiers.quantization import QuantizationModifier; print('llmcompressor: OK')" 2>/dev/null || echo "llmcompressor: optional/skipped") \
        && (python -c "import gptqmodel; print('GPTQModel: OK')" 2>/dev/null || echo "GPTQModel: optional/skipped") \
        && (python -c "import bitsandbytes; print(f'BitsAndBytes: OK')" 2>/dev/null || echo "BitsAndBytes: optional/skipped") \
        && python -c "import lm_eval; print('lm-eval: OK')"; \
    else \
        python -c "import torch; print(f'[基础环境] PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"; \
    fi

# =============================================================================
# 健康检查
# =============================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" || exit 1

# =============================================================================
# 入口点
# =============================================================================
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["bash"]
