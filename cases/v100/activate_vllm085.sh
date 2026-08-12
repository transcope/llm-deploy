#!/usr/bin/env bash
# 激活部署评测环境 (vllm-venv, vLLM 0.8.5)
# 用于 V100 上部署/评测 Qwen3 GPTQ 模型，无量化工具链
# 这是最新落地方案 (vllm 0.8.5 支持 Qwen3 + V100)
source /app/vllm-venv/bin/activate
echo "✅ 部署评测环境 (vllm-venv, vLLM 0.8.5)"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo N/A)"
echo "  vllm:      $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo N/A)"
