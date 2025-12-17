#!/bin/bash
# 224 + EVA02 + SNE(backbone-ot, prior=prob+softunion, cos, mean) + Patch-FPN + pi 监督 - Road3D（无 Prompt）

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road.py"

echo "=========================================="
echo "[F2c+pi-mean-cos-prob-softunion-road] 224 + EVA02 + SNE(backbone-ot, prior=prob+softunion, cos, mean) + Patch-FPN + pi 监督 - Road3D (NoPrompt)"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2c+pi-mean-cos-prob-softunion-road] 训练完成!"
echo "=========================================="
