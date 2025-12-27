#!/bin/bash
# 消融实验: Learnable Only 基线 (ORFD 数据集)
# - 512 + EVA02 + NoSNE + learnable_only
# - 仅使用 2 个可学习掩码向量，无文本编码
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_learnable_only.py"

echo "=========================================="
echo "[Learnable Only - ORFD] 512 + EVA02 + learnable_only"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "说明: 仅可学习掩码向量，无文本编码器 (基线)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[Learnable Only - ORFD] 训练完成!"
echo "=========================================="
