#!/bin/bash

set -e

# 配置日志
LOG_DIR='./logs'
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/quantization_${TIMESTAMP}.log"
MAX_RETRIES=3
RETRY_DELAY=60

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# 检查输入参数
if [ $# -eq 0 ]; then
    log "错误：未指定量化类型（普通/GPTQ）"
    exit 1
fi

type=$1  # 获取第一个参数作为量化类型

# 根据类型选择脚本
case $type in
    "normal")
        SCRIPT="awq_quantity.py"
        log "启动普通量化 (${SCRIPT})..."
        ;;
    "gptq")
        SCRIPT="gptq_quantity.py"
        log "启动GPTQ量化 (${SCRIPT})..."
        ;;
    *)
        log "错误：未知的量化类型 '$type'（可选: normal/gptq）"
        exit 1
        ;;
esac

# 带重试机制的脚本执行
retry=0
while [ $retry -lt $MAX_RETRIES ]; do
    if python3 "$SCRIPT"; then
        log "${SCRIPT} 执行成功"
        exit 0
    else
        retry=$((retry + 1))
        log "警告：${SCRIPT} 执行失败 (尝试 $retry/$MAX_RETRIES)，${RETRY_DELAY}秒后重试..."
        sleep $RETRY_DELAY
    fi
done

log "错误：${SCRIPT} 超过最大重试次数 ($MAX_RETRIES)"
exit 1