#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) + TopK - ORFD

GPU=0
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_learnableT_promptTau_topk.py"

echo "=========================================="
echo "[F2pSoft-config-topk-ORFD] 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, config, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) + TopK(10/3) - ORFD"
echo "GPU: $GPU, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

CUDA_VISIBLE_DEVICES=$GPU python train.py \
    --config $CONFIG \
    --gpu-id=0 \
    --seed=$SEED \
    --deterministic

echo "=========================================="
echo "[F2pSoft-config-topk-ORFD] 训练完成!"
echo "=========================================="
