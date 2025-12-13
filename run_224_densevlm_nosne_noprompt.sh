#!/bin/bash
# 消融实验 B2: 224 + DenseVLM + NoSNE + NoPrompt (预训练权重对比)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_densevlm_nosne_noprompt.py"

echo "=========================================="
echo "[B2] 224 + DenseVLM + NoSNE + NoPrompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[B2] 训练完成!"
echo "=========================================="
