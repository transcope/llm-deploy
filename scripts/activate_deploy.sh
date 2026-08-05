#!/usr/bin/env bash
# 激活部署评测环境 (venv-deploy)
# 用于 vLLM 部署和模型评测，无量化工具链
source /app/venv-deploy/bin/activate
echo "✅ 部署评测环境 (venv-deploy)"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo N/A)"
echo "  vllm:      $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
