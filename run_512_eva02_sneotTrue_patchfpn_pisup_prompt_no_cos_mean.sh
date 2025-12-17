#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 监督 + Prompt - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean.py"

echo "=========================================="
echo "[G2p-mean-cos] 512 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 监督 + Prompt - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[G2p-mean-cos] 训练完成!"
echo "=========================================="
