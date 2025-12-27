#!/bin/bash
# P6: SNE OT + Patch-FPN + Text Pretrain + PromptCls (text=T, img=F) - Road3D
GPU=${1:-0}
SEED=42
echo "[P6-Road] SNE OT + Patch-FPN + Text Pretrain + PromptCls"
CONFIG="configs/ablation_road/exp_512_eva02_sneot_patchfpn_text_pretrain_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
