#!/bin/bash
# 消融实验 E1: 512 + DenseVLM + SNE(backbone-proj) + Prompt (完整模型)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_densevlm_sne_prompt_road.py"

echo "=========================================="
echo "[E1] 512 + DenseVLM + SNE(backbone-proj) + Prompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[E1] 训练完成!"
echo "=========================================="
