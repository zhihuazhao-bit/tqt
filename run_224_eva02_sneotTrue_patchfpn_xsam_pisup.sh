#!/bin/bash
# 消融实验 F2d-xsam: 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN(PixelSampling) + pi 深监督 - ORFD
# 默认单卡训练，更新 GPU/SEED 即可。

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt.py"

echo "=========================================="
echo "[F2d-xsam] 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN(PixelSampling) + pi 监督 - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "对比: F2d(conv版 Patch-FPN) vs F2d-xsam(PixelSampling版 Patch-FPN)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2d-xsam] 训练完成!"
echo "=========================================="
