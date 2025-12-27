#!/bin/bash
# P6: SNE OT + Patch-FPN + Text Pretrain + PromptCls (text=T, img=F) - ORFD
GPU=${1:-0}
SEED=42
echo "[P6-ORFD] SNE OT + Patch-FPN + Text Pretrain + PromptCls"
CONFIG="configs/ablation/exp_512_eva02_sneot_patchfpn_text_pretrain.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
