#!/bin/bash
# P3: SNE Proj + Full Pretrain (text=T, img=T, prompt_cls=F) - Road3D
GPU=${1:-0}
SEED=42
echo "[P3-Road] SNE Proj + Full Pretrain"
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_m2f_full_pretrain_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
