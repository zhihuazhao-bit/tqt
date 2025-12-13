#!/bin/bash
# 消融实验 F2c+pi: 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN + pi 深监督 - ORFD
# 默认单卡训练，更新 GPU/SEED 即可。

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt.py"

echo "=========================================="
echo "[F2c+pi] 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN + pi 监督 - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "对比: F2c(无 pi 监督) vs F2c+pi(有 pi 监督)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2c+pi] 训练完成!"
echo "=========================================="
