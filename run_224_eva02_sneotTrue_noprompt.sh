#!/bin/bash
# 消融实验 F2b: 224 + EVA02 + SNE(backbone-ot) + NoPrompt (ORFD)
# OT Prior: True (使用预测图分配文本分布权重)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_noprompt.py"

echo "=========================================="
echo "[F2b] 224 + EVA02 + SNE(backbone-ot, prior=True) + NoPrompt - ORFD"
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
echo "[F2b] 训练完成!"
echo "=========================================="
