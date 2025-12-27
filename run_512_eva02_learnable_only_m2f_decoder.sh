#!/bin/bash
# 消融实验: Learnable Only + 标准 Mask2Former Pixel Decoder (ORFD 数据集)
# - 512 + EVA02 + NoSNE + learnable_only
# - 使用标准 MSDeformAttnPixelDecoder (无文本交叉注意力)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_learnable_only_m2f_decoder.py"

echo "=========================================="
echo "[Learnable Only + M2F Decoder - ORFD] 512 + EVA02 + learnable_only + MSDeformAttnPixelDecoder"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "说明: 仅可学习掩码向量 + 标准 Mask2Former pixel decoder (无文本交叉注意力)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[Learnable Only + M2F Decoder - ORFD] 训练完成!"
echo "=========================================="
