#!/bin/bash
# 消融实验 F2c: 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN + NoPrompt (ORFD)
# 对比 F2b (patch_fpn=False) 观察 patch FPN 对特征金字塔的影响
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_noprompt.py"

echo "=========================================="
echo "[F2c] 224 + EVA02 + SNE(backbone-ot, prior=True) + Patch-FPN + NoPrompt - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "对比: F2b (patch_fpn=False)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2c] 训练完成!"
echo "=========================================="
