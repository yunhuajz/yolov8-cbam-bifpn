#!/bin/bash
# -*- coding: utf-8 -*-
# YOLOv8 批量训练脚本 - 性能优化版
# ============================================

set -e  # 遇到错误立即退出

# ========== 清理残留进程 ==========
echo "[*] 清理残留 Python 进程..."
pkill -9 python 2>/dev/null || true
sleep 2

# ========== 设置环境变量 ==========
# 获取脚本所在目录（支持空格）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
cd "$SCRIPT_DIR"

echo "[*] 项目目录: $SCRIPT_DIR"
echo "[*] PYTHONPATH: $PYTHONPATH"

# ========== 检查必要文件 ==========
if [ ! -f "scripts/train.py" ]; then
    echo "[✗] 错误: 未找到 scripts/train.py"
    exit 1
fi

# ========== 自动备份旧结果 ==========
RUNS_DIR="$SCRIPT_DIR/runs/train"
if [ -d "$RUNS_DIR" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    BACKUP_DIR="$SCRIPT_DIR/runs/train_backup_${TIMESTAMP}"
    echo "[*] 备份旧训练结果到: $BACKUP_DIR"
    mv "$RUNS_DIR" "$BACKUP_DIR"
fi

# ========== 训练参数优化 ==========
# 4090D 跑 YOLOv8n 系列，32 是非常健康的 Batch Size
BATCH=32  

echo ""
echo "========================================"
echo "  YOLOv8 批量训练 (性能优化版)"
echo "========================================"
echo "  Batch: $BATCH"
echo "  Epochs: 100"
echo "  Device: cuda:0"
echo "========================================"
echo ""

# ========== 实验列表 ==========
declare -a EXPERIMENTS=(
    "configs/expA_baseline.yaml:expA_baseline"
    "configs/expB_cbam.yaml:expB_cbam"
    "configs/expC_bifpn.yaml:expC_bifpn"
    "configs/expD_combined.yaml:expD_combined"
)

TOTAL=${#EXPERIMENTS[@]}
CURRENT=0

# ========== 执行训练 ==========
for exp in "${EXPERIMENTS[@]}"; do
    CURRENT=$((CURRENT + 1))
    CONFIG_FILE="${exp%%:*}"
    EXP_NAME="${exp##*:}"

    echo ""
    echo "========================================"
    echo "  [$CURRENT/$TOTAL] $EXP_NAME"
    echo "========================================"

    # 运行训练脚本
    python scripts/train.py \
        --config "$CONFIG_FILE" \
        --name "$EXP_NAME" \
        --batch $BATCH

    if [ $? -eq 0 ]; then
        echo "[✓] $EXP_NAME 完成"
    else
        echo "[✗] $EXP_NAME 失败"
        exit 1
    fi

    # 实验间短暂暂停，释放显存
    if [ $CURRENT -lt $TOTAL ]; then
        echo "[*] 等待 5 秒后启动下一个实验..."
        sleep 5
    fi
done

echo ""
echo "========================================"
echo "  所有实验完成!"
echo "========================================"