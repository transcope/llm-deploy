#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${PROJECT_ROOT}/vllm-env"
CONTAINER_VENV="/app/venv"

# 优先使用项目本地虚拟环境 (本地开发); 否则复用容器内镜像内置 venv (服务器)
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
elif [ -d "${CONTAINER_VENV}" ]; then
    source "${CONTAINER_VENV}/bin/activate"
else
    echo "未找到虚拟环境: 本地请先执行 ./init 创建 vllm-env; 服务器请确认容器内 ${CONTAINER_VENV} 存在"
    exit 1
fi

echo "运行测试..."
python -m pytest "${PROJECT_ROOT}/tests" -v
