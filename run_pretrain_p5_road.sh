#!/bin/bash
# P5: SNE Proj + Text Pretrain + PromptCls (text=T, img=F, prompt_cls=T) - Road3D
GPU=${1:-0}
SEED=42
echo "[P5-Road] SNE Proj + Text Pretrain + PromptCls"
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
