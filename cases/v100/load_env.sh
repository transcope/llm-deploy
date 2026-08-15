#!/usr/bin/env bash
# =============================================================================
# 加载 V100 服务器连接环境变量 (脱敏)
#
# 用法:
#   source cases/v100/load_env.sh
#
# 说明:
#   - 从 configs/.env 读取真实连接信息
#   - 若 .env 不存在, 提示先复制 .env.example
#   - 加载后可用 $V100_HOST, $V100_USER, $V100_CONTAINER 等变量
# =============================================================================

ENV_FILE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../configs" && pwd)/.env"

if [ ! -f "$ENV_FILE" ]; then
    echo -e "\033[0;31m错误: 未找到 $ENV_FILE\033[0m"
    echo -e "\033[1;33m请先执行: cp configs/.env.example configs/.env 并填入真实值\033[0m"
    return 1 2>/dev/null || exit 1
fi

# 加载 .env (忽略注释和空行)
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

echo -e "\033[0;32m已加载 V100 连接信息:\033[0m"
echo -e "  SSH:   ${V100_USER}@${V100_HOST}:${V100_PORT}"
echo -e "  容器:  ${V100_CONTAINER}"
echo -e "  工作目录: ${V100_WORKDIR}"
