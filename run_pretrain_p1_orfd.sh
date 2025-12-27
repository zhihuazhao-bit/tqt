#!/bin/bash
# P1: SNE Proj + Learnable Only + No Pretrain (text=F, img=F) - ORFD
GPU=${1:-0}
SEED=42
echo "[P1-ORFD] SNE Proj + Learnable Only + No Pretrain"
CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_no_pretrain.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
