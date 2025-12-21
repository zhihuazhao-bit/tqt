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

# F2p-mean-cos: 224 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_cos="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251215_1728/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_cos="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road.py"

# F2p-mean-cos-512: 512 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_cos_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/20251216_1157/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road/best_mIoU_iter_3000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_cos_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_road.py"

# F2p-mean-cos-soft-512: 512 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup + Prompt(Soft)
CKPT_F2pSoft_mean_cos_soft_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road/20251217_1300/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road/best_mIoU_iter_3000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_mean_cos_soft_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_road.py"

# F2p-mean-prob-soft: 224 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_prob_soft="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_prob_soft="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road.py"

# F2p-mean-prob-soft-512: 512 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup + Prompt
CKPT_F2p_mean_prob_soft_512="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/20251216_1157/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2p_mean_prob_soft_512="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road.py"

# F2pSoft-learnableT-promptTau-224: 224 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, Tau)
CKPT_F2pSoft_learnableT_promptTau_224="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251220_0944/exp_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_5000.pth"  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_224="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2pSoft-learnableT-promptTau-1024: 1024 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, Tau)
CKPT_F2pSoft_learnableT_promptTau_1024=""  # TODO: 填入 Road3D checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau_1024="configs/ablation_road/exp_1024_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2p-mean-prob-soft-0.5-512: 512 + EVA02 + SNE(OT, prior=prob+softunion, T=0.5, cos, mean) + Patch-FPN + piSup + Prompt(Soft)
CKPT_F2pSoft_mean_prob_soft_512_T05="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5/20251217_1304/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_mean_prob_soft_512_T05="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_0.5.py"

# F2p-mean-prob-soft-0.5-512-hard: 512 + EVA02 + SNE(OT, prior=prob+softunion, T=0.5, cos, mean) + Patch-FPN + piSup + Prompt(Hard)
CKPT_F2pHard_mean_prob_soft_512_T05="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5/20251217_1304/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2pHard_mean_prob_soft_512_T05="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_prompt_no_cos_mean_prob_softunion_road_0.5.py"

# F2c+pi-mean-cos: 224 + EVA02 + SNE(OT, prior=True, cos, mean) + Patch-FPN + piSup (NoPrompt)
CKPT_F2c_pi_mean_cos="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/20251215_1727/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c_pi_mean_cos="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_road.py"

# F2c+pi-mean-cos-prob-softunion: 224 + EVA02 + SNE(OT, prior=prob+softunion, cos, mean) + Patch-FPN + piSup (NoPrompt)
CKPT_F2c_pi_mean_prob_soft="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/20251215_1726/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road/best_mIoU_iter_5000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c_pi_mean_prob_soft="configs/ablation_road/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_no_cos_mean_prob_softunion_road.py"

# F2pSoft-learnableT-promptTau: 512 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, mean, softunion) + Patch-FPN + piSup + Prompt(Soft, uses Tau)
CKPT_F2pSoft_learnableT_promptTau_1="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1243/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth" # TODO: 填入 checkpoint 路径
CKPT_F2pSoft_learnableT_promptTau_0_1="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1320/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth" # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_learnableT_promptTau="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# F2pSoft-proj-learnableT-promptTau: 512 + EVA02 + SNE(OT, prior=prob, Learnable T, cos, proj, softunion) + Patch-FPN + piSup + Prompt(Soft, uses Tau)
CKPT_F2pSoft_proj_learnableT_promptTau="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau/20251218_1243/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_1000.pth" # TODO: 填入 checkpoint 路径
CONFIG_F2pSoft_proj_learnableT_promptTau="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_proj_prob_softunion_road_learnableT_promptTau.py"

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
# run_test "F2p-mean-cos" "$CONFIG_F2p_mean_cos" "$CKPT_F2p_mean_cos"
# run_test "F2p-mean-cos-512" "$CONFIG_F2p_mean_cos_512" "$CKPT_F2p_mean_cos_512"
# run_test "F2p-mean-cos-soft-512" "$CONFIG_F2p_mean_cos_soft_512" "$CKPT_F2p_mean_cos_soft_512"
# run_test "F2p-mean-prob-soft" "$CONFIG_F2p_mean_prob_soft" "$CKPT_F2p_mean_prob_soft"
# run_test "F2p-mean-prob-soft-512" "$CONFIG_F2p_mean_prob_soft_512" "$CKPT_F2p_mean_prob_soft_512"
# run_test "F2p-mean-prob-soft-0.5-512" "$CONFIG_F2p_mean_prob_soft_512_T05" "$CKPT_F2p_mean_prob_soft_512_T05"
# run_test "F2p-mean-prob-soft-0.5-512-hard" "$CONFIG_F2p_mean_prob_soft_512_T05_hard" "$CKPT_F2p_mean_prob_soft_512_T05_hard"
# run_test "F2pSoft-learnableT-promptTau-224" "$CONFIG_F2pSoft_learnableT_promptTau_224" "$CKPT_F2pSoft_learnableT_promptTau_224"
# run_test "F2c+pi-mean-cos" "$CONFIG_F2c_pi_mean_cos" "$CKPT_F2c_pi_mean_cos"
# run_test "F2c+pi-mean-cos-prob-softunion" "$CONFIG_F2c_pi_mean_prob_soft" "$CKPT_F2c_pi_mean_prob_soft"
# run_test "F2pSoft-mean-prob-soft-0.5-512" "$CONFIG_F2pSoft_mean_prob_soft_512_T05" "$CKPT_F2pSoft_mean_prob_soft_512_T05"
# run_test "F2pHard-mean-prob-soft-0.5-512" "$CONFIG_F2pHard_mean_prob_soft_512_T05" "$CKPT_F2pHard_mean_prob_soft_512_T05"
# run_test "F2p-mean-cos-soft-512" "$CONFIG_F2pSoft_mean_cos_soft_512" "$CKPT_F2pSoft_mean_cos_soft_512"
# run_test "F2pSoft-learnableT-promptTau" "$CONFIG_F2pSoft_learnableT_promptTau" "$CKPT_F2pSoft_learnableT_promptTau_1"
# run_test "F2pSoft-learnableT-promptTau" "$CONFIG_F2pSoft_learnableT_promptTau" "$CKPT_F2pSoft_learnableT_promptTau_0_1"
# run_test "F2pSoft-proj-learnableT-promptTau" "$CONFIG_F2pSoft_proj_learnableT_promptTau" "$CKPT_F2pSoft_proj_learnableT_promptTau"
## 1024 分辨率
# run_test "F2pSoft-learnableT-promptTau-1024" "$CONFIG_F2pSoft_learnableT_promptTau_1024" "$CKPT_F2pSoft_learnableT_promptTau_1024"
run_test "F2pSoft-learnableT-promptTau_224" "$CONFIG_F2pSoft_learnableT_promptTau_224" "$CKPT_F2pSoft_learnableT_promptTau_224"


echo ""
echo "=========================================="
echo "所有测试完成!"
echo ""
echo "CSV 文件位置 (在各自 checkpoint 目录的 test_results/ 下):"
echo "  查找命令: find ./work_dirs -name 'testing_eval_file_stats_*.csv'"
echo ""
echo "统计结果: python statis_ablation_results.py"
echo "=========================================="
