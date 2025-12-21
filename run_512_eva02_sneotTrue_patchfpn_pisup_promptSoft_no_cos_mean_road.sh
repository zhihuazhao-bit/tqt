#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 监督 + Prompt(Soft) - Road3D

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road.py"

echo "=========================================="
echo "[F2p-mean-cos-soft-road] 512 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 监督 + Prompt(Soft) - Road3D"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2p-mean-cos-soft-road] 训练完成!"
echo "=========================================="
