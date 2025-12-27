#!/bin/bash
# 消融实验: Material Classification (ORFD 数据集)
# - 512 + EVA02 + SNE(OT) + Patch-FPN + piSup + Prompt(Soft, Tau) + Material
# - 增加材质分类，场景库从24扩展到240
# GPU: 0

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau_material.py"

echo "=========================================="
echo "[Material Classification - ORFD] 512 + EVA02 + Material"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "说明: 增加材质分类 (场景库 24 -> 240)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[Material Classification - ORFD] 训练完成!"
echo "=========================================="
