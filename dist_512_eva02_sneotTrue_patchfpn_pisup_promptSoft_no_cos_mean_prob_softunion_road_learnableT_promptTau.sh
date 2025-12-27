#!/bin/bash
# 512 + EVA02 + SNE(backbone-ot, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft, uses Tau) - Road3D
# 并行训练版本

GPUS=4
SEED=42
CONFIG="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

echo "=========================================="
echo "[F2p-mean-prob-soft-0.5-road] 512 + EVA02 + SNE(backbone-ot, prior=prob, T=0.5, cos, mean, softunion) + Patch-FPN + pi 监督 + Prompt(Soft) - Road3D"
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
echo "[F2p-mean-prob-soft-0.5-road] 训练完成!"
echo "=========================================="
