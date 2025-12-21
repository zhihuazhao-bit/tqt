#!/bin/bash
# 1024 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_1024_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

echo "=========================================="
echo "[F2pSoft-learnableT-promptTau-Road-1024] 1024 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2pSoft-learnableT-promptTau-Road-1024] 训练完成!"
echo "=========================================="
