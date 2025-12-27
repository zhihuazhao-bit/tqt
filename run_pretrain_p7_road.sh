#!/bin/bash
# P7: SNE OT + Patch-FPN + Text Pretrain + No PromptCls (text=T, img=F, prompt_cls=F) - Road3D
GPU=${1:-0}
SEED=42
echo "[P7-Road] SNE OT + Patch-FPN + Text Pretrain + No PromptCls"
CONFIG="configs/ablation_road/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
