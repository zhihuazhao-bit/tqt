#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, linear+text cls) - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau_linear_text.py"

echo "=========================================="
echo "[F2pSoft-linear_text-learnableT-promptTau-ORFD] 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, linear+text cls) - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2pSoft-linear_text-learnableT-promptTau-ORFD] 训练完成!"
echo "=========================================="
