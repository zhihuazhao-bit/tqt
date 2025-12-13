#!/bin/bash
# 消融实验测试脚本 - 串行运行所有测试
# 
# 使用方法:
#   1. 修改下方 CKPT_* 变量，填入各实验的 checkpoint 路径
#   2. 运行: bash test_all_ablation.sh
#   3. 测试完成后，运行 python statis_ablation_results.py 统计结果

GPU=0

# ============================================================================
# 请在此处填入各实验的 checkpoint 路径
# ============================================================================

# A1: 224 + EVA02 + NoSNE + NoPrompt
CKPT_A1="/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt/20251203_1719/exp_224_eva02_nosne_noprompt/best_mIoU_iter_1000.pth"
CONFIG_A1="configs/ablation/exp_224_eva02_nosne_noprompt.py"

# A1-nc: 224 + EVA02 + NoSNE + NoPrompt + NoContextDecoder
CKPT_A1NC="/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_nocd/20251212_1823/exp_224_eva02_nosne_noprompt_nocd/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_A1NC="configs/ablation/exp_224_eva02_nosne_noprompt_nocd.py"

# A2: 512 + EVA02 + NoSNE + NoPrompt  
CKPT_A2="/root/tqdm/work_dirs/ablation_512_eva02_nosne_noprompt/20251203_1719/exp_512_eva02_nosne_noprompt/best_mIoU_iter_1000.pth"
CONFIG_A2="configs/ablation/exp_512_eva02_nosne_noprompt.py"

# B2: 224 + DenseVLM + NoSNE + NoPrompt
CKPT_B2="/root/tqdm/work_dirs/ablation_224_densevlm_nosne_noprompt/20251203_1717/exp_224_densevlm_nosne_noprompt/best_mIoU_iter_1000.pth"
CONFIG_B2="configs/ablation/exp_224_densevlm_nosne_noprompt.py"

# C2: 224 + DenseVLM + SNE + NoPrompt
CKPT_C2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_noprompt/20251203_1718/exp_224_densevlm_sne_noprompt/best_mIoU_iter_1000.pth"
CONFIG_C2="configs/ablation/exp_224_densevlm_sne_noprompt.py"

# D2: 224 + DenseVLM + SNE + Prompt
CKPT_D2="/root/tqdm/work_dirs/ablation_224_densevlm_sne_prompt/20251203_1718/exp_224_densevlm_sne_prompt/best_mIoU_iter_1000.pth"
CONFIG_D2="configs/ablation/exp_224_densevlm_sne_prompt.py"

# E1: 512 + DenseVLM + SNE + Prompt
CKPT_E1="/root/tqdm/work_dirs/ablation_512_densevlm_sne_prompt/20251203_1719/exp_512_densevlm_sne_prompt/best_mIoU_iter_1000.pth"
CONFIG_E1="configs/ablation/exp_512_densevlm_sne_prompt.py"

# E2: 512 + DenseVLM + NoSNE + Prompt
CKPT_E2="/root/tqdm/work_dirs/ablation_512_densevlm_nosne_prompt/20251204_0815/exp_512_densevlm_nosne_prompt/iter_1000.pth"
CONFIG_E2="configs/ablation/exp_512_densevlm_nosne_prompt.py"

# F1a: 224 + DenseVLM + SNE(OT, prior=False) + NoPrompt
CKPT_F1a="/root/tqdm/work_dirs/ablation_224_densevlm_sneot_noprompt/20251204_1638/exp_224_densevlm_sneot_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F1a="configs/ablation/exp_224_densevlm_sneotFalse_noprompt.py"

# F1b: 224 + DenseVLM + SNE(OT, prior=True) + NoPrompt
CKPT_F1b="/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_noprompt/20251204_1724/exp_224_densevlm_sneotTrue_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F1b="configs/ablation/exp_224_densevlm_sneotTrue_noprompt.py"

# F2a: 224 + EVA02 + SNE(OT, prior=False) + NoPrompt
CKPT_F2a="/root/tqdm/work_dirs/ablation_224_eva02_sneot_noprompt/20251204_1638/exp_224_eva02_sneot_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2a="configs/ablation/exp_224_eva02_sneotFalse_noprompt.py"

# F2b: 224 + EVA02 + SNE(OT, prior=True) + NoPrompt
CKPT_F2b="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_noprompt/20251204_1727/exp_224_eva02_sneotTrue_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2b="configs/ablation/exp_224_eva02_sneotTrue_noprompt.py"

# F2c: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN + NoPrompt
CKPT_F2c="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_noprompt/20251207_1303/exp_224_eva02_sneotTrue_patchfpn_noprompt/iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_noprompt.py"

# F2d: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN + piSup + NoPrompt
CKPT_F2d="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt/20251208_0812/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt.py"

CKPT_F2d1="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt/20251212_2239/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d1="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt.py"

CKPT_F2d2="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-l2/20251212_2322/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d2="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_L2.py"

CKPT_F2d3="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-no-l2/20251212_2257/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d3="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_No_L2.py"

# F2d-xsam: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN(PixelSampling) + piSup + NoPrompt
CKPT_F2dXSAM="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/20251212_1850/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2dXSAM="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt.py"

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
# run_test "A1-nc" "$CONFIG_A1NC" "$CKPT_A1NC"  # 无 context decoder
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
# run_test "F2c" "$CONFIG_F2c" "$CKPT_F2c"
# run_test "F2d" "$CONFIG_F2d" "$CKPT_F2d"  # pi 监督版 Patch-FPN（补齐 CKPT 后取消注释）
run_test "F2d1" "$CONFIG_F2d1" "$CKPT_F2d1"
run_test "F2d2" "$CONFIG_F2d2" "$CKPT_F2d2"
run_test "F2d3" "$CONFIG_F2d3" "$CKPT_F2d3"
# run_test "F2d-xsam" "$CONFIG_F2dXSAM" "$CKPT_F2dXSAM"  # PixelSampling Patch-FPN + pi 监督
# run_test "F2d" "$CONFIG_F2d" "$CKPT_F2d"  # 默认先跑 conv 版，替换 CKPT 后任选

echo ""
echo "=========================================="
echo "所有测试完成!"
echo ""
echo "CSV 文件位置 (在各自 checkpoint 目录的 test_results/ 下):"
echo "  查找命令: find ./work_dirs -name 'testing_eval_file_stats_*.csv'"
echo ""
echo "统计结果: python statis_ablation_results.py"
echo "=========================================="
