#!/bin/bash
# 并行启动所有6个消融实验 (使用6张GPU)
#
# 实验分配:
#   GPU 0: A1 - 224 + EVA02 + NoSNE + NoPrompt
#   GPU 1: A2 - 512 + EVA02 + NoSNE + NoPrompt
#   GPU 2: B2 - 224 + DenseVLM + NoSNE + NoPrompt
#   GPU 3: C2 - 224 + DenseVLM + SNE + NoPrompt
#   GPU 4: D2 - 224 + DenseVLM + SNE + Prompt
#   GPU 5: E1 - 512 + DenseVLM + SNE + Prompt

echo "=========================================="
echo "并行启动 6 个消融实验"
echo "=========================================="
echo ""
echo "GPU 0: A1 - 224 + EVA02 + NoSNE + NoPrompt"
echo "GPU 1: A2 - 512 + EVA02 + NoSNE + NoPrompt"
echo "GPU 2: B2 - 224 + DenseVLM + NoSNE + NoPrompt"
echo "GPU 3: C2 - 224 + DenseVLM + SNE + NoPrompt"
echo "GPU 4: D2 - 224 + DenseVLM + SNE + Prompt"
echo "GPU 5: E1 - 512 + DenseVLM + SNE + Prompt"
echo ""
echo "=========================================="

# 创建日志目录
mkdir -p logs/ablation

# 并行启动所有实验
nohup bash run_224_eva02_nosne_noprompt.sh > logs/ablation/A1_224_eva02_nosne_noprompt.log 2>&1 &
echo "已启动 A1 (GPU 0), PID: $!"

nohup bash run_512_eva02_nosne_noprompt.sh > logs/ablation/A2_512_eva02_nosne_noprompt.log 2>&1 &
echo "已启动 A2 (GPU 1), PID: $!"

nohup bash run_224_densevlm_nosne_noprompt.sh > logs/ablation/B2_224_densevlm_nosne_noprompt.log 2>&1 &
echo "已启动 B2 (GPU 2), PID: $!"

nohup bash run_224_densevlm_sne_noprompt.sh > logs/ablation/C2_224_densevlm_sne_noprompt.log 2>&1 &
echo "已启动 C2 (GPU 3), PID: $!"

nohup bash run_224_densevlm_sne_prompt.sh > logs/ablation/D2_224_densevlm_sne_prompt.log 2>&1 &
echo "已启动 D2 (GPU 4), PID: $!"

nohup bash run_512_densevlm_sne_prompt.sh > logs/ablation/E1_512_densevlm_sne_prompt.log 2>&1 &
echo "已启动 E1 (GPU 5), PID: $!"

echo ""
echo "=========================================="
echo "所有实验已在后台启动!"
echo ""
echo "查看日志:"
echo "  tail -f logs/ablation/A1_224_eva02_nosne_noprompt.log"
echo "  tail -f logs/ablation/A2_512_eva02_nosne_noprompt.log"
echo "  tail -f logs/ablation/B2_224_densevlm_nosne_noprompt.log"
echo "  tail -f logs/ablation/C2_224_densevlm_sne_noprompt.log"
echo "  tail -f logs/ablation/D2_224_densevlm_sne_prompt.log"
echo "  tail -f logs/ablation/E1_512_densevlm_sne_prompt.log"
echo ""
echo "查看 GPU 使用: nvidia-smi"
echo "=========================================="
