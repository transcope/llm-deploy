#!/usr/bin/env bash
# 激活量化环境 (venv-quant)
# 用于模型量化，含 gptqmodel 2.0.0 等量化工具链
source /app/venv-quant/bin/activate
echo "✅ 量化环境 (venv-quant)"
echo "  gptqmodel: $(python -c 'import gptqmodel; print(gptqmodel.__version__)' 2>/dev/null || echo N/A)"
echo "  torch:     $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo N/A)"
