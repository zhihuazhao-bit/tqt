#!/bin/bash
# P7: SNE OT + Patch-FPN + Text Pretrain + No PromptCls (text=T, img=F, prompt_cls=F) - ORFD
GPU=${1:-0}
SEED=42
echo "[P7-ORFD] SNE OT + Patch-FPN + Text Pretrain + No PromptCls"
CONFIG="configs/ablation/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls.py"
CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
