#!/bin/bash
# 消融实验 A1: 224 + EVA02 + NoSNE + NoPrompt (基准-小尺寸)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_224_eva02_nosne_noprompt_road.py"

echo "=========================================="
echo "[A1] 224 + EVA02 + NoSNE + NoPrompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[A1] 训练完成!"
echo "=========================================="
