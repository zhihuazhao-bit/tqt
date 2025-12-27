#!/bin/bash
# P5: SNE Proj + Text Pretrain + PromptCls (text=T, img=F, prompt_cls=T) - ORFD
GPU=${1:-0}
SEED=42
echo "[P5-ORFD] SNE Proj + Text Pretrain + PromptCls"
CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
