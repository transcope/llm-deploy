#!/usr/bin/env bash
# 激活部署评测环境 (venv-deploy, 旧版 vllm 0.7.1)
# ⚠️ 旧环境：vllm 0.7.1 不支持 Qwen3 架构，仅用于非 Qwen3 模型。
# 最新落地方案请用 activate_vllm085.sh (vllm 0.8.5)。
source /app/venv-deploy/bin/activate
echo "✅ 部署评测环境 (venv-deploy, vllm 0.7.1) [旧]"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo N/A)"
echo "  vllm:      $(python -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo N/A)"
