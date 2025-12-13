#!/bin/bash
# 消融实验 F2a: 224 + EVA02 + SNE(backbone-ot) + NoPrompt (Road3D)
# OT Prior: False (均匀分布)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_224_eva02_sneotFalse_noprompt_road.py"

echo "=========================================="
echo "[F2a] 224 + EVA02 + SNE(backbone-ot, prior=False) + NoPrompt - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "对比: F2a (prior=False) vs F2b (prior=True)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2a] 训练完成!"
echo "=========================================="
