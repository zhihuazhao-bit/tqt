#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Fixed T=0.01, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_fixedT0.01_promptTau.py"

echo "=========================================="
echo "[F2pSoft-fixedT0.01-promptTau-ORFD] 512 + EVA02 + SNE(backbone-ot, prior=prob, Fixed T=0.01, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD"
echo "OT Temperature: Fixed 0.01"
echo "Tau: Learnable"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2pSoft-fixedT0.01-promptTau-ORFD] 训练完成!"
echo "=========================================="
