#!/bin/bash
# 消融实验 A1-nc: 224 + EVA02 + NoSNE + NoPrompt + NoContextDecoder
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_nosne_noprompt_nocd.py"

echo "=========================================="
echo "[A1-nc] 224 + EVA02 + NoSNE + NoPrompt + NoContextDecoder"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[A1-nc] 训练完成!"
echo "=========================================="
