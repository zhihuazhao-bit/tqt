#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, T=0.5, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft) - Road3D

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5.py"

echo "=========================================="
echo "[F2p-mean-prob-soft-0.5-road] 512 + EVA02 + SNE(backbone-ot, prior=prob, T=0.5, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft) - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2p-mean-prob-soft-0.5-road] 训练完成!"
echo "=========================================="
