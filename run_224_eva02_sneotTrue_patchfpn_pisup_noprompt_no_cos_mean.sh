#!/bin/bash
# 消融实验 F2c+pi(mean, cos): 224 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 深监督 - ORFD
# 默认单卡训练，更新 GPU/SEED 即可。

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean.py"

echo "=========================================="
echo "[F2c+pi-mean-cos] 224 + EVA02 + SNE(backbone-ot, prior=True, cos, mean) + Patch-FPN + pi 监督 - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "对比: F2c+pi(proj, cos) vs F2c+pi(mean, cos)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2c+pi-mean-cos] 训练完成!"
echo "=========================================="
