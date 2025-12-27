#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD
# 并行训练版本

GPUS=4
SEED=42
CONFIG="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau.py"

echo "=========================================="
echo "[F2pSoft-learnableT-promptTau-ORFD] 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - ORFD"
echo "GPUs: $GPUS, Seed: $SEED"
echo "Config: $CONFIG"
echo "=========================================="

python3 -m torch.distributed.launch \
    --nproc_per_node=$GPUS \
    train.py \
    --config $CONFIG \
    --launcher pytorch \
    --gpus $GPUS \
    --seed $SEED \
    --deterministic

wait

echo "=========================================="
echo "[F2pSoft-learnableT-promptTau-ORFD] 训练完成!"
echo "=========================================="
