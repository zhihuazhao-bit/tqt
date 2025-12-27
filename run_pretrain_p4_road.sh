#!/bin/bash
# P4: SNE Proj + Full Pretrain + PromptCls (text=T, img=T, prompt_cls=T) - Road3D
GPU=${1:-0}
SEED=42
echo "[P4-Road] SNE Proj + Full Pretrain + PromptCls"
CONFIG="configs/ablation_road/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
