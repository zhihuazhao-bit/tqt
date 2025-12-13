#!/bin/bash
# 消融实验 E1: 512 + DenseVLM + SNE(backbone-proj) + Prompt (完整模型)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_densevlm_nosne_prompt.py"

echo "=========================================="
echo "[E2] 512 + DenseVLM + NoSNE(backbone-proj) + Prompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[E2] 训练完成!"
echo "=========================================="
