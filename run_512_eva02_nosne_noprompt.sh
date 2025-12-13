#!/bin/bash
# 消融实验 A2: 512 + EVA02 + NoSNE + NoPrompt (基准-大尺寸)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_nosne_noprompt.py"

echo "=========================================="
echo "[A2] 512 + EVA02 + NoSNE + NoPrompt"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[A2] 训练完成!"
echo "=========================================="
