#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD
# 注意：不使用 TopK，保持原始行为

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_learnableT_promptTau.py"

echo "=========================================="
echo "[F2pSoft-config-ORFD] 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD"
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
echo "[F2pSoft-config-ORFD] 训练完成!"
echo "=========================================="
