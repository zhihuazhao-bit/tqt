#!/bin/bash
# 512 + EVA02 + SNE(backbone-cross_attn, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + Prompt(Soft, uses Tau) - Road3D

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneCrossAttnTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

echo "=========================================="
echo "[CrossAttn-learnableT-promptTau-Road3D] 512 + EVA02 + SNE(backbone-cross_attn) + Patch-FPN + Prompt(Soft, uses Tau) - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[CrossAttn-learnableT-promptTau-Road3D] 训练完成!"
echo "=========================================="
