#!/bin/bash
# P2: SNE Proj + Text Pretrain Only (text=T, img=F, prompt_cls=F) - ORFD
GPU=${1:-0}
SEED=42
echo "[P2-ORFD] SNE Proj + Text Pretrain Only"
CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_text_pretrain.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
