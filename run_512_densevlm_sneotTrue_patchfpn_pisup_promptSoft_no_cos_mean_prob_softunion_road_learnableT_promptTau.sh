#!/bin/bash
# 512 + DenseVLM + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_densevlm_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

echo "=========================================="
echo "[DenseVLM-learnableT-promptTau-Road3D] 512 + DenseVLM + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[DenseVLM-learnableT-promptTau-Road3D] 训练完成!"
echo "=========================================="
