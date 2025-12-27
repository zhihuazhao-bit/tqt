#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D
# 注意：不使用 TopK，保持原始行为

GPU=0
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau.py"

echo "=========================================="
echo "[F2pSoft-config-Road3D] 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D"
echo "TopK: None (不使用)"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2pSoft-config-Road3D] 训练完成!"
echo "=========================================="
