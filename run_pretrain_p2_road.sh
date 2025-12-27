#!/bin/bash
# P2: SNE Proj + Text Pretrain Only (text=T, img=F, prompt_cls=F) - Road3D
GPU=${1:-0}
SEED=42
echo "[P2-Road] SNE Proj + Text Pretrain Only"
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_m2f_text_pretrain_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
