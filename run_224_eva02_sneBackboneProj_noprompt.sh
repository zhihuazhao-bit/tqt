#!/bin/bash
# 224 + EVA02 + SNE(backbone-proj) + NoPrompt - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneBackboneProj_noprompt.py"

echo "=========================================="
echo "[A1-backbone-proj] 224 + EVA02 + SNE(backbone-proj) + NoPrompt - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[A1-backbone-proj] 训练完成!"
echo "=========================================="
