#!/bin/bash
# 消融实验: Learnable Only + SNE Proj 融合 + 标准 Mask2Former Pixel Decoder (Road3D 数据集)
# - 512 + EVA02 + SNE(proj) + learnable_only + MSDeformAttnPixelDecoder
# - 使用 SNE proj 融合，在 backbone 阶段融合表面法线特征
# - 使用标准 MSDeformAttnPixelDecoder (无文本交叉注意力)
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_learnable_only_m2f_decoder_road.py"

echo "=========================================="
echo "[SNE-Proj + Learnable Only + M2F Decoder - Road3D] 512 + EVA02 + SNE(proj) + learnable_only + MSDeformAttnPixelDecoder"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "说明: SNE proj 融合 + 仅可学习掩码向量 + 标准 Mask2Former pixel decoder (无文本交叉注意力)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[SNE-Proj + Learnable Only + M2F Decoder - Road3D] 训练完成!"
echo "=========================================="
