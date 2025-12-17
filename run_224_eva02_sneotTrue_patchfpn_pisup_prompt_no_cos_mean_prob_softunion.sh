#!/bin/bash
# 224 + EVA02 + SNE(backbone-ot, prior=True, cos, mean+prob, softunion) + Patch-FPN + pi 监督 + Prompt - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion.py"

echo "=========================================="
echo "[F2p-mean-prob-soft] 224 + EVA02 + SNE(backbone-ot, prior=True, cos, mean+prob, softunion) + Patch-FPN + pi 监督 + Prompt - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2p-mean-prob-soft] 训练完成!"
echo "=========================================="
