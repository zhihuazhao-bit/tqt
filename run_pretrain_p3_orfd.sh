#!/bin/bash
# P3: SNE Proj + Full Pretrain (text=T, img=T, prompt_cls=F) - ORFD
GPU=${1:-0}
SEED=42
echo "[P3-ORFD] SNE Proj + Full Pretrain"
CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_full_pretrain.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
