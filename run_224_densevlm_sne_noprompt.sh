#!/bin/bash
# 消融实验 C2: 224 + DenseVLM + SNE(backbone-proj) + NoPrompt (+SNE)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_densevlm_sne_noprompt.py"

echo "=========================================="
echo "[C2] 224 + DenseVLM + SNE(backbone-proj) + NoPrompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[C2] 训练完成!"
echo "=========================================="
