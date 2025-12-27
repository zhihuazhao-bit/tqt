#!/bin/bash
# 消融实验: SNE-OT + Learnable Only (Road3D 数据集)
# - 512 + EVA02 + SNE(OT) + Patch-FPN + piSup + learnable_only
# - 保留完整OT架构，仅用可学习向量替代文本编码
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_learnable_only_road.py"

echo "=========================================="
echo "[SNE-OT + Learnable Only - Road3D] 512 + EVA02 + SNE(OT) + learnable_only"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "说明: 保留OT架构，用可学习向量替代文本编码器"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[SNE-OT + Learnable Only - Road3D] 训练完成!"
echo "=========================================="
