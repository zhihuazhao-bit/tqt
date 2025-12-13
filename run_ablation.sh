#!/bin/bash
# TQT 消融实验批量运行脚本
# 
# 实验设计:
#   A1: 224 + EVA02 + NoSNE + NoPrompt  (基准-小尺寸)
#   A2: 512 + EVA02 + NoSNE + NoPrompt  (基准-大尺寸)
#   B2: 224 + DenseVLM + NoSNE + NoPrompt  (预训练对比)
#   C2: 224 + DenseVLM + SNE + NoPrompt  (+SNE)
#   D2: 224 + DenseVLM + SNE + Prompt  (+Prompt)
#   E1: 512 + DenseVLM + SNE + Prompt  (完整模型)

set -e  # 出错即停止

GPU=${1:-0}
SEED=42

echo "=========================================="
echo "TQT 消融实验"
echo "GPU: $GPU, Seed: $SEED"
echo "=========================================="

# 实验 A1: 基准-小尺寸 (224 + EVA02)
echo ""
echo "[A1] 224 + EVA02 + NoSNE + NoPrompt"
python train.py --config configs/ablation/exp_224_eva02_nosne_noprompt.py --gpu-id=$GPU --seed=$SEED --deterministic

# 实验 A2: 基准-大尺寸 (512 + EVA02)
echo ""
echo "[A2] 512 + EVA02 + NoSNE + NoPrompt"
python train.py --config configs/ablation/exp_512_eva02_nosne_noprompt.py --gpu-id=$GPU --seed=$SEED --deterministic

# 实验 B2: 预训练对比 (224 + DenseVLM)
echo ""
echo "[B2] 224 + DenseVLM + NoSNE + NoPrompt"
python train.py --config configs/ablation/exp_224_densevlm_nosne_noprompt.py --gpu-id=$GPU --seed=$SEED --deterministic

# 实验 C2: +SNE (backbone-proj)
echo ""
echo "[C2] 224 + DenseVLM + SNE(backbone-proj) + NoPrompt"
python train.py --config configs/ablation/exp_224_densevlm_sne_noprompt.py --gpu-id=$GPU --seed=$SEED --deterministic

# 实验 D2: +Prompt
echo ""
echo "[D2] 224 + DenseVLM + SNE + Prompt"
python train.py --config configs/ablation/exp_224_densevlm_sne_prompt.py --gpu-id=$GPU --seed=$SEED --deterministic

# 实验 E1: 完整模型 (512)
echo ""
echo "[E1] 512 + DenseVLM + SNE + Prompt"
python train.py --config configs/ablation/exp_512_densevlm_sne_prompt.py --gpu-id=$GPU --seed=$SEED --deterministic

echo ""
echo "=========================================="
echo "所有消融实验完成!"
echo "=========================================="
