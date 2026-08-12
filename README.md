# 大模型推理压缩与本地化部署方案

[![GitHub](https://img.shields.io/badge/GitHub-transcope%2Fllm--deploy-181717?logo=github)](https://github.com/transcope/llm-deploy)

面向服务器 GPU 环境的大模型量化压缩与高效推理部署完整解决方案，支持 Qwen/DeepSeek 系列 7B-32B+ 模型及多模态 VLM。

## 核心特性

- **多种量化方案**: AWQ (推荐)、FP8 (H100+)、GPTQ、SmoothQuant W8A8
- **高效推理引擎**: vLLM V1 引擎 (PagedAttention + Continuous Batching)
- **标准化接口**: OpenAI-compatible API，零成本迁移
- **完整评测体系**: lm-eval 精度评测 + 吞吐/延迟性能测试
- **多模态支持**: Qwen2.5-VL / DeepSeek-VL 图文推理
- **即开即用**: 一键量化、一键部署、一键评测

## 快速开始

### 1. 环境准备

```bash
# 一键初始化（创建虚拟环境、安装依赖、创建目录）
./init

# 激活环境
source vllm-env/bin/activate  # Linux/Mac
# vllm-env\Scripts\activate  # Windows
```

`./init` 会自动尝试安装部署评测环境依赖（`requirements-vllm085.txt`，vllm 0.8.5）；
若本地为 macOS / 无 CUDA / Python 版本不兼容，会提示在服务器 CUDA 环境安装。

服务器部署时（Docker 容器内），所有任务使用同一基础环境：

```bash
# 激活容器基础环境（已含 torch + vLLM + 量化工具链）
source /app/venv/bin/activate
```

> **物理 V100 服务器**（非 Docker）因 gptqmodel 2.0.0 与 optimum 版本冲突需使用双虚拟环境隔离，
> 详见 [docs/V100_SERVER_GUIDE.md](docs/V100_SERVER_GUIDE.md) 第 3 节。Docker 用户无需关心。

**硬件要求:**
- NVIDIA GPU (计算能力 >= 7.0；V100 为 7.0，需使用 float16 + GPTQ，见 V100 指南)
- CUDA >= 12.1
- 显存: 7B模型>=16GB, 32B模型>=40GB

### 2. 模型量化

	```bash
	# AWQ INT4 量化 (推荐，Ampere+ 通用GPU)
	python llm_deploy/quantize_model.py \
	    --model Qwen/Qwen2.5-7B-Instruct \
	    --method awq \
	    --output ./models/Qwen2.5-7B-AWQ

	# GPTQ INT4 量化 (V100 推荐，EXL2 kernel 支持 SM 7.0)
	python llm_deploy/quantize_model.py \
	    --model Qwen/Qwen2.5-7B-Instruct \
	    --method gptq \
	    --config configs/gptq_4bit.yaml \
	    --output ./models/Qwen2.5-7B-GPTQ

	# W8A8 SmoothQuant 量化 (V100 可用，精度损失最小)
	python llm_deploy/quantize_model.py \
	    --model Qwen/Qwen2.5-7B-Instruct \
	    --method w8a8 \
	    --config configs/w8a8.yaml \
	    --output ./models/Qwen2.5-7B-W8A8

	# BitsAndBytes NF4 无需预量化，部署时加 --quantization bitsandbytes 即可

	# FP8 量化 (H100/H200/B200，V100 不支持)
	python llm_deploy/quantize_model.py \
	    --model Qwen/Qwen2.5-7B-Instruct \
	    --method fp8 \
	    --output ./models/Qwen2.5-7B-FP8
	```

> **V100 用户注意**: 请使用 `--method gptq` (推荐) 或 `w8a8` / `bitsandbytes`，
> 不要使用 `awq` (无高速 kernel) 和 `fp8` (硬件不支持)。
> 完整说明见 [docs/V100_DEPLOY_GUIDE.md](docs/V100_DEPLOY_GUIDE.md) 第 4 节。

### 3. 启动推理服务

```bash
# 单卡部署 AWQ 模型
python llm_deploy/deploy_server.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --quantization awq

# 或使用 vllm 命令直接启动
vllm serve ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.9
```

服务启动后，访问 http://localhost:8000 获取 OpenAI 兼容 API。

**V100 + Qwen3 部署**（vLLM 0.8.5，`vllm-venv`）：

```bash
# vLLM 0.7.1 不支持 Qwen3，V100 上需用 vLLM 0.8.5
source /app/vllm-venv/bin/activate
python llm_deploy/serve_vllm085.py \
    --model /volume/models/Mind-SLLM-Qwen3-8B-GPTQ \
    --quantization gptq --port 8000 --gpu 0
```

> V100 上 vLLM 0.8.5 推理速度约 **30 tok/s**，比 gptqmodel TORCH backend（2.6 tok/s）快约 **11.5 倍**。
> 详见 [docs/V100_DEPLOY_GUIDE.md](docs/V100_DEPLOY_GUIDE.md) 4.5 节。

### 4. 测试验证

```bash
# 测试服务
 curl http://localhost:8000/v1/models

# 对话测试
 curl http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
     "model": "Qwen/Qwen2.5-7B-Instruct",
     "messages": [{"role": "user", "content": "你好，请介绍一下自己"}]
   }'
```

### 5. 评测验证

```bash
# 标准 Benchmark 精度评测 (GSM8K / HellaSwag 等)
python llm_deploy/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --quantization awq \
    --tasks gsm8k,hellaswag \
    --baseline-model Qwen/Qwen2.5-7B-Instruct

# 领域精度评测 (从领域数据构建的 custom Benchmark)
python llm_deploy/benchmark_domain.py \
    --base-url http://localhost:8000 \
    --model Qwen2.5-7B-AWQ

# 构建领域精度 Benchmark 数据集 (从 data/custom_data/ 自动提取)
python llm_deploy/build_accuracy_benchmark.py --num-samples 300

# 性能测试 (需服务已启动)
python llm_deploy/benchmark_eval.py \
    --model ./models/Qwen2.5-7B-AWQ \
    --perf-test \
    --num-prompts 100 \
    --concurrency 10
```

> 领域精度评测使用 `data/custom_data/` 中的领域数据构建 QA Benchmark，
> 通过关键词召回率衡量模型在通信/数学/代码等领域的实际业务能力。
> 详见 [评估协议](docs/EVALUATION_PROTOCOL.md)。

### 6. 本地测试

```bash
# 运行全部单元测试（无需 GPU，使用 mock 验证脚本逻辑）
./run_tests.sh

# 或直接运行
source vllm-env/bin/activate
python -m pytest tests/ -v
```

### 7. 完整使用示例

`cases/` 目录提供了按硬件分类的服务器端到端命令和应用例：

```bash
cases/a100/07_a100_deploy.sh          # A100 单卡端到端 (量化→部署→评测) ★
cases/v100/serve_gptq.py              # V100 回退部署服务 (gptqmodel TORCH backend)
cases/v100/compare_models.py          # V100 回退对比评测 (原模型 vs 量化模型)
```

> 完整使用方式（量化/评测/部署命令模板、按 GPU 选方案）见 [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md)

## 场景化部署指南

### 7B 模型单卡部署

```bash
# Qwen2.5-7B AWQ 单卡部署 (RTX 4090 24GB / A100 40GB+)
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code
```

### A100 单卡端到端 (量化→部署→评测)

A100 (SM 8.0) 原生支持 bfloat16 与 AWQ GEMM kernel，是 AWQ INT4 的首选平台。一键脚本：

```bash
# 一键全流程: AWQ 量化 + 精度评测
./examples/07_a100_deploy.sh all

# 或分阶段执行
./examples/07_a100_deploy.sh quantize                    # 1. AWQ 量化
./examples/07_a100_deploy.sh deploy ./models/Qwen2.5-7B-AWQ  # 2. 启动服务
./examples/07_a100_deploy.sh eval ./models/Qwen2.5-7B-AWQ    # 3. 精度评测
./examples/07_a100_deploy.sh perf                          # 4. 性能测试 (另开终端)
```

> 完整说明见 [docs/A100_DEPLOY_GUIDE.md](docs/A100_DEPLOY_GUIDE.md)

### 32B 模型多卡部署

```bash
# Qwen2.5-32B 双卡张量并行 (2x A100 40GB)
vllm serve Qwen/Qwen2.5-32B-Instruct \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --trust-remote-code
```

### DeepSeek-R1 蒸馏模型部署

```bash
# DeepSeek-R1-Distill-Qwen-14B AWQ
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B-AWQ \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.92 \
    --max-model-len 8192 \
    --trust-remote-code \
    --tool-call-parser deepseek_v3
```

### 多模态模型部署

```bash
# Qwen2.5-VL-7B 图文推理
vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
    --trust-remote-code \
    --limit-mm-per-prompt image=5 \
    --max-model-len 32768
```

## V100 服务器 Docker 部署

针对 **8卡 V100 32GB** 服务器提供完整 Docker 环境与一键部署脚本：

```bash
cd docker

# 1. 构建镜像
docker-compose build

# 2. 启动容器
docker-compose up -d vllm-server
docker exec -it vllm-v100 bash

# 3. 一键部署模型
./v100-deploy.sh qwen2.5-7b        # 7B 单卡 (FP16)
./v100-deploy.sh qwen2.5-7b-gptq   # 7B GPTQ INT4 单卡 (V100 推荐)
./v100-deploy.sh qwen2.5-32b       # 32B 双卡
./v100-deploy.sh qwen2.5-32b-gptq  # 32B GPTQ INT4 单卡
./v100-deploy.sh deepseek-r1-14b   # DeepSeek 14B
./v100-deploy.sh --list            # 查看所有支持模型
```

> V100 (SM 7.0) 限制：**不支持 FP8、AWQ GEMM kernel、FlashAttention-2**  
> V100 推荐量化：**GPTQ INT4** / **BitsAndBytes NF4** / **SmoothQuant W8A8**  
> 详见 [docs/V100_DEPLOY_GUIDE.md](docs/V100_DEPLOY_GUIDE.md) 与 [docs/GPU_ARCHITECTURE_GUIDE.md](docs/GPU_ARCHITECTURE_GUIDE.md)

## 项目结构

```
		llm-deploy/
		├── init                         # 一键初始化脚本
		├── run_tests.sh                 # 测试入口
		├── llm_deploy/                         # Python 核心代码
		│   ├── quantize_model.py       # 模型量化转换
		│   ├── deploy_server.py        # vLLM 服务部署
		│   ├── serve_vllm085.py        # vLLM 0.8.5 部署服务 (V100+Qwen3 推荐)
		│   ├── benchmark_eval.py       # 评测与性能测试
		│   ├── benchmark_domain.py     # 领域精度评测 (API/本地模式, vllm 0.8.5)
		│   ├── hf_download.py          # HuggingFace 模型下载
		│   ├── qwen3_gptq_adapter.py   # Qwen3 GPTQ 适配器 (gptqmodel MODEL_MAP 注入)
		│   ├── qwen3_pipeline_patch.py # Qwen3 llmcompressor pipeline 兼容补丁
		│   ├── build_accuracy_benchmark.py  # 领域精度 Benchmark 构建
		│   ├── build_calibration_data.py    # 校准/评估数据集构建
		│   └── validate_calibration.py      # 量化后 PPL 验证
		├── cases/                      # 执行脚本 + 应用例 (按硬件组织)
		│   ├── v100/
		│   │   ├── activate_quant.sh       # V100 量化环境快捷激活
		│   │   ├── activate_vllm085.sh     # V100 部署评测环境快捷激活 (vllm 0.8.5)
		│   │   ├── activate_deploy.sh      # V100 旧部署环境快捷激活 (vllm 0.7.1)
		│   │   ├── install_quant_tools.sh  # V100 量化工具链安装 (Dockerfile 调用)
		│   │   ├── serve_gptq.py           # V100 回退部署服务 (gptqmodel TORCH backend)
		│   │   └── compare_models.py       # V100 回退对比评测 (原模型 vs 量化模型)
		│   └── a100/
		│       └── 07_a100_deploy.sh       # A100 单卡端到端 (量化→部署→评测)
		├── configs/                     # 配置文件模板
		│   ├── awq_4bit.yaml           # AWQ INT4 (A100+ 首选)
		│   ├── fp8.yaml                # FP8 (H100+)
		│   ├── gptq_4bit.yaml          # GPTQ INT4 (通用)
		│   ├── gptq_4bit_v100.yaml     # GPTQ V100 (llmcompressor 后端, A100+ 部署)
		│   ├── gptq_4bit_v100_gptqmodel.yaml  # GPTQ V100 (gptqmodel 后端, V100 生产推荐)
		│   ├── w8a8.yaml               # SmoothQuant W8A8
		│   ├── bitsandbytes_nf4.yaml   # NF4 动态量化 (免预量化)
		│   └── vllm_serve.yaml         # vLLM 服务默认配置
		├── docker/                      # V100 Docker 部署
		│   ├── Dockerfile
		│   ├── docker-compose.yml
		│   ├── entrypoint.sh
		│   └── v100-deploy.sh
		├── tests/                       # 单元测试（mock，无 GPU 可运行）
		├── docs/                        # 文档
		│   ├── USAGE_GUIDE.md           # 使用指南 (量化/评测/部署 + GPU适配) ★导读入口
		│   ├── CALIBRATION_GUIDE.md     # 校准数据指南
		│   ├── EVALUATION_PROTOCOL.md   # 评估协议 (PPL + 领域精度评测)
		│   ├── V100_SERVER_GUIDE.md     # V100 服务器操作指南 (SSH/Docker/双虚拟环境)
		│   ├── V100_DEPLOY_GUIDE.md     # V100 部署专版
		│   ├── A100_DEPLOY_GUIDE.md     # A100 单卡部署专版
		│   └── GPU_ARCHITECTURE_GUIDE.md # GPU 架构兼容性指南
		├── bak/                         # 历史版本存档 (gitignore)
		├── data/                        # 领域数据 (gitignore)
		├── models/                      # 量化模型存放 (gitignore)
		├── results/                     # 评测结果 (gitignore)
		├── cache/                       # HuggingFace 缓存 (gitignore)
		├── logs/                        # 日志 (gitignore)
		├── requirements-quant.txt       # 量化环境依赖快照
		├── requirements-vllm085.txt     # 部署评测环境依赖快照 (vllm 0.8.5)
		└── README.md                    # 本文件
	```
	└── README.md                    # 本文件
```

## 文档导航

- **[从零执行操作手册](docs/FROM_SCRATCH_RUNBOOK.md)** —— 清空环境后从零完成压缩→部署→评估全链路（8 步）
- **[从零执行测试报告](docs/FROM_SCRATCH_TEST_REPORT.md)** —— 实际执行结果、精度对比、问题记录
- **[使用指南](docs/USAGE_GUIDE.md)** —— 量化/评测/部署总览 + 按 GPU 选方案（推荐先读）
- **[校准数据指南](docs/CALIBRATION_GUIDE.md)** —— 校准样本数、数据格式、离线校准
- **[V100 服务器操作指南](docs/V100_SERVER_GUIDE.md)** —— SSH/Docker/多虚拟环境操作（venv-quant + vllm-venv）
- **[V100 部署指南](docs/V100_DEPLOY_GUIDE.md)** —— V100 专版（GPTQ 双后端、vLLM 0.8.5 部署、显存调参、Docker）
- **[A100 部署指南](docs/A100_DEPLOY_GUIDE.md)** —— A100 单卡端到端（AWQ 量化、一键脚本）
- **[GPU 架构兼容性指南](docs/GPU_ARCHITECTURE_GUIDE.md)** —— V100/A100/H100 跨硬件迁移

## 量化方案对比

| 方案 | 显存节省 | 精度保留 | 硬件要求 | 推荐场景 |
|------|----------|----------|----------|----------|
| **AWQ INT4** | 75% | ~95% | Ampere+ | 通用GPU生产部署 |
| **FP8** | 50% | ~99% | Hopper+ (H100+) | H100+最佳性能 |
| **GPTQ INT4** | 75% | ~90% | Turing+ | 兼容性优先 |
| **W8A8 INT8** | 50% | ~96% | 通用 | 精度敏感场景 |

## 实测精度损失（Mind-SLLM-Qwen3-8B, V100, GPTQ INT4, vLLM 0.8.5）

基于 **86 样本领域精度 Benchmark**（通信/数学/代码混合数据集）的实测对比（vLLM 0.8.5 本地评测）：

| 指标 | 原模型 (FP16) | GPTQ 4-bit | **精度损失** |
|------|:------------:|:----------:|:----------:|
| **总体准确率** | **43.02%** | **39.53%** | **-3.49%** |

> GPTQ 4-bit 量化保留约 **92%** 基线精度（39.53/43.02），math 类任务无损失。
> 模型体积从 **16 GB → 5.8 GB**（-63%），推理速度从 2.6 tok/s（TORCH backend）提升到
> **30 tok/s**（vLLM 0.8.5，快 11.5 倍）。完整评测数据见 `results/vllm085/baseline.json` 和
> `results/vllm085/gptq.json`。

## 相关资源

- [vLLM 官方文档](https://docs.vllm.ai/)
- [llm-compressor GitHub](https://github.com/vllm-project/llm-compressor)
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- [Qwen 官方文档](https://qwen.readthedocs.io/)
- [DeepSeek 官方 GitHub](https://github.com/deepseek-ai/)

## License

MIT License
