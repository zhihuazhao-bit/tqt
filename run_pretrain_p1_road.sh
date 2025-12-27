#!/bin/bash
# P1: SNE Proj + Learnable Only + No Pretrain (text=F, img=F) - Road3D
GPU=${1:-0}
SEED=42
echo "[P1-Road] SNE Proj + Learnable Only + No Pretrain"
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_m2f_no_pretrain_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
