#!/bin/bash
# 预训练权重消融实验 - ORFD 数据集
# 包含8个实验配置

GPU=0
SEED=42

echo "=========================================="
echo "预训练权重消融实验 - ORFD"
echo "=========================================="

# 实验 #1: SNE Proj + Learnable Only + No Pretrain
run_exp1() {
    echo "[#1] SNE Proj + Learnable Only + No Pretrain"
    CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_no_pretrain.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #2: SNE Proj + Text Pretrain Only
run_exp2() {
    echo "[#2] SNE Proj + Text Pretrain Only"
    CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_text_pretrain.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #3: SNE Proj + Full Pretrain
run_exp3() {
    echo "[#3] SNE Proj + Full Pretrain"
    CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_full_pretrain.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #4: SNE Proj + Full Pretrain + PromptCls
run_exp4() {
    echo "[#4] SNE Proj + Full Pretrain + PromptCls"
    CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #5: SNE Proj + Text Pretrain + PromptCls
run_exp5() {
    echo "[#5] SNE Proj + Text Pretrain + PromptCls"
    CONFIG="configs/ablation/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #6: SNE OT + Patch-FPN + Text Pretrain + PromptCls
run_exp6() {
    echo "[#6] SNE OT + Patch-FPN + Text Pretrain + PromptCls"
    CONFIG="configs/ablation/exp_512_eva02_sneot_patchfpn_text_pretrain.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #7: SNE OT + Patch-FPN + Text Pretrain + No PromptCls
run_exp7() {
    echo "[#7] SNE OT + Patch-FPN + Text Pretrain + No PromptCls"
    CONFIG="configs/ablation/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 实验 #8: SNE OT + Patch-FPN + Full Pretrain + No PromptCls
run_exp8() {
    echo "[#8] SNE OT + Patch-FPN + Full Pretrain + No PromptCls"
    CONFIG="configs/ablation/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls.py"
    CUDA_VISIBLE_DEVICES=$GPU python train.py --config $CONFIG --gpu-id=0 --seed=$SEED --deterministic
}

# 运行所有实验 (取消注释需要运行的实验)
run_exp1
run_exp2
run_exp3
run_exp4
run_exp5
run_exp6
run_exp7
run_exp8

echo "=========================================="
echo "所有 ORFD 预训练消融实验完成!"
echo "=========================================="
