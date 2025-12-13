#!/bin/bash
# 消融实验测试脚本 (Road3D 数据集) - 串行运行所有测试
# 
# 使用方法:
#   1. 修改下方 CKPT_* 变量，填入各实验的 checkpoint 路径
#   2. 运行: bash test_all_ablation_road.sh
#   3. 测试完成后，运行 python statis_ablation_results.py 统计结果

GPU=0

# ============================================================================
# 请在此处填入各实验的 checkpoint 路径 (Road3D 数据集)
# ============================================================================

# A1: 224 + EVA02 + NoSNE + NoPrompt
CKPT_A1="/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_road/20251203_2312/exp_224_eva02_nosne_noprompt_road/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_A1="configs/ablation_road/exp_224_eva02_nosne_noprompt_road.py"

# A2: 512 + EVA02 + NoSNE + NoPrompt  
CKPT_A2="/root/tqdm/work_dirs/ablation_512_eva02_nosne_noprompt_road/20251203_2313/exp_512_eva02_nosne_noprompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_A2="configs/ablation_road/exp_512_eva02_nosne_noprompt_road.py"

# B2: 224 + DenseVLM + NoSNE + NoPrompt
CKPT_B2="/root/tqdm/work_dirs/ablation_224_densevlm_nosne_noprompt_road/20251203_2311/exp_224_densevlm_nosne_noprompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_B2="configs/ablation_road/exp_224_densevlm_nosne_noprompt_road.py"

# C2: 224 + DenseVLM + SNE + NoPrompt
CKPT_C2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_noprompt_road/20251203_2311/exp_224_densevlm_sne_noprompt_road/best_mIoU_iter_2000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_C2="configs/ablation_road/exp_224_densevlm_sne_noprompt_road.py"

# D2: 224 + DenseVLM + SNE + Prompt ## 
CKPT_D2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_prompt_road/20251203_2312/exp_224_densevlm_sne_prompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_D2="configs/ablation_road/exp_224_densevlm_sne_prompt_road.py"

# E1: 512 + DenseVLM + SNE + Prompt ##
CKPT_E1="/root/tqdm/work_dirs/ablation_512_densevlm_sne_prompt_road/20251203_2312/exp_512_densevlm_sne_prompt_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_E1="configs/ablation_road/exp_512_densevlm_sne_prompt_road.py"

# E2: 512 + DenseVLM + NoSNE + Prompt
CKPT_E2=""  # TODO: 填入 checkpoint 路径
CONFIG_E2="configs/ablation_road/exp_512_densevlm_nosne_prompt_road.py"

# F1a: 224 + DenseVLM + SNE(OT, prior=False) + NoPrompt
CKPT_F1a=""  # TODO: 填入 checkpoint 路径
CONFIG_F1a="configs/ablation_road/exp_224_densevlm_sneotFalse_noprompt_road.py"

# F1b: 224 + DenseVLM + SNE(OT, prior=True) + NoPrompt
CKPT_F1b=""  # TODO: 填入 checkpoint 路径
CONFIG_F1b="configs/ablation_road/exp_224_densevlm_sneotTrue_noprompt_road.py"

# F2a: 224 + EVA02 + SNE(OT, prior=False) + NoPrompt
CKPT_F2a=""  # TODO: 填入 checkpoint 路径
CONFIG_F2a="configs/ablation_road/exp_224_eva02_sneotFalse_noprompt_road.py"

# F2b: 224 + EVA02 + SNE(OT, prior=True) + NoPrompt
CKPT_F2b=""  # TODO: 填入 checkpoint 路径
CONFIG_F2b="configs/ablation_road/exp_224_eva02_sneotTrue_noprompt_road.py"

# ============================================================================

echo "=========================================="
echo "消融实验测试 - 串行运行"
echo "GPU: $GPU"
echo "=========================================="

# 测试函数
run_test() {
    local EXP_ID=$1
    local CONFIG=$2
    local CHECKPOINT=$3
    
    # 自动从 checkpoint 路径获取保存目录
    local SAVE_DIR=$(dirname "$CHECKPOINT")/test_results/
    
    echo ""
    echo "=========================================="
    echo "[$EXP_ID] 开始测试"
    echo "Config: $CONFIG"
    echo "Checkpoint: $CHECKPOINT"
    echo "Save Dir: $SAVE_DIR"
    echo "=========================================="
    
    if [ ! -f "$CHECKPOINT" ]; then
        echo "警告: Checkpoint 文件不存在: $CHECKPOINT, 跳过此实验"
        return 1
    fi
    
    CUDA_VISIBLE_DEVICES=$GPU python test.py \
        --config $CONFIG \
        --checkpoint $CHECKPOINT \
        --eval mIoU mFscore \
        --show-dir $SAVE_DIR \
        --save_dir $SAVE_DIR
    
    echo "[$EXP_ID] 测试完成!"
    echo "CSV 保存位置: $SAVE_DIR"
}

# 串行运行所有测试
# run_test "A1" "$CONFIG_A1" "$CKPT_A1"
# run_test "A2" "$CONFIG_A2" "$CKPT_A2"
# run_test "B2" "$CONFIG_B2" "$CKPT_B2"
# run_test "C2" "$CONFIG_C2" "$CKPT_C2"
# run_test "D2" "$CONFIG_D2" "$CKPT_D2"
# run_test "E1" "$CONFIG_E1" "$CKPT_E1"
# run_test "E2" "$CONFIG_E2" "$CKPT_E2"
# run_test "F1a" "$CONFIG_F1a" "$CKPT_F1a"
# run_test "F1b" "$CONFIG_F1b" "$CKPT_F1b"
# run_test "F2a" "$CONFIG_F2a" "$CKPT_F2a"
# run_test "F2b" "$CONFIG_F2b" "$CKPT_F2b"

echo ""
echo "=========================================="
echo "所有测试完成!"
echo ""
echo "CSV 文件位置 (在各自 checkpoint 目录的 test_results/ 下):"
echo "  查找命令: find ./work_dirs -name 'testing_eval_file_stats_*.csv'"
echo ""
echo "统计结果: python statis_ablation_results.py"
echo "=========================================="
