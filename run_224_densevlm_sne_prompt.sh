#!/bin/bash
# 消融实验 D2: 224 + DenseVLM + SNE(backbone-proj) + Prompt (+Prompt)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_densevlm_sne_prompt.py"

echo "=========================================="
echo "[D2] 224 + DenseVLM + SNE(backbone-proj) + Prompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[D2] 训练完成!"
echo "=========================================="
