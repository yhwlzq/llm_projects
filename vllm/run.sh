#!/bin/bash

# 进入脚本所在目录
cd "$(dirname "$0")"

# 配置
HOST="0.0.0.0"
PORT=8000
WORKERS=1  # 开发时建议先用1个worker
LOG_LEVEL="debug"
TIMEOUT=120
APP_MODULE="server:app"  # 明确指定模块和app变量名

# 启动命令
echo "Starting server at http://${HOST}:${PORT}"
uvicorn ${APP_MODULE} \
    --host ${HOST} \
    --port ${PORT} \
    --workers ${WORKERS} \
    --log-level ${LOG_LEVEL} \
    --reload