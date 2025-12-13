#!/bin/bash
# 消融实验推理脚本 - 串行运行详细可视化/中间结果导出
# 参考 test_all_ablation.sh，调用 inference_detailed.py 生成可视化、指标与可选的中间张量。
# 
# 使用方法:
#   1. 填写各实验的 CONFIG 与 CKPT 路径。
#   2. 按需设置 DEBUG/ATTN 开关与输出目录。
#   3. 运行: bash inference_all_ablation.sh

GPU=0

# 是否保存中间结果 (score map / OT pi) 与注意力矩阵
SAVE_DEBUG=true
SAVE_ATTN=true
# 中间结果保存根目录（按 config 名称自动分子目录）
DEBUG_DIR="./work_dirs/infer_debug"
# 可视化输出根目录（按 config 名称自动分子目录）
VIS_DIR="./work_dirs/detailed_inference"

# 最大推理样本数 (-1 表示全部)
LIMIT=-1
# 是否显示进度条
PROGRESS=true

# ============================================================================
# 在此填入各实验的 config / checkpoint
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
CKPT_F1a="/root/tqdm/work_dirs/ablation_224_densevlm_sneot_noprompt/20251204_1638/exp_224_densevlm_sneot_noprompt/best_mIoU_iter_1000.pth"
CONFIG_F1a="configs/ablation/exp_224_densevlm_sneotFalse_noprompt.py"

# F1b: 224 + DenseVLM + SNE(OT, prior=True) + NoPrompt
CKPT_F1b="/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_noprompt/20251204_1724/exp_224_densevlm_sneotTrue_noprompt/best_mIoU_iter_1000.pth"
CONFIG_F1b="configs/ablation/exp_224_densevlm_sneotTrue_noprompt.py"

# F2a: 224 + EVA02 + SNE(OT, prior=False) + NoPrompt
CKPT_F2a="/root/tqdm/work_dirs/ablation_224_eva02_sneot_noprompt/20251204_1638/exp_224_eva02_sneot_noprompt/best_mIoU_iter_1000.pth"
CONFIG_F2a="configs/ablation/exp_224_eva02_sneotFalse_noprompt.py"

# F2b: 224 + EVA02 + SNE(OT, prior=True) + NoPrompt
CKPT_F2b="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_noprompt/20251204_1727/exp_224_eva02_sneotTrue_noprompt/best_mIoU_iter_1000.pth"
CONFIG_F2b="configs/ablation/exp_224_eva02_sneotTrue_noprompt.py"

# F2c: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN + NoPrompt
CKPT_F2c="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_noprompt/20251207_1303/exp_224_eva02_sneotTrue_patchfpn_noprompt/iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2c="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_noprompt.py"

# F2d: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN + NoPrompt
CKPT_F2d="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_old/20251208_0812/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt.py"


CKPT_F2d1="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt/20251212_2239/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d1="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt.py"

CKPT_F2d2="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-l2/20251212_2322/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_l2/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d2="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_L2.py"

CKPT_F2d3="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-no-l2/20251212_2257/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt-no-l2/best_mIoU_iter_4000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2d3="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_No_L2.py"

# F2d-xsam: 224 + EVA02 + SNE(OT, prior=True) + Patch-FPN(PixelSampling) + piSup + NoPrompt
CKPT_F2dXSAM="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/20251212_1850/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填入 checkpoint 路径
CONFIG_F2dXSAM="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt.py"

# ============================================================================

echo "=========================================="
echo "消融实验推理 - 串行运行"
echo "GPU: $GPU"
echo "=========================================="

doit() {
    local EXP_ID=$1
    local CONFIG=$2
    local CHECKPOINT=$3

    if [ ! -f "$CHECKPOINT" ]; then
        echo "警告: Checkpoint 文件不存在: $CHECKPOINT, 跳过 $EXP_ID"
        return 1
    fi

    local SAVE_DIR="$VIS_DIR/$(basename ${CONFIG%.*})"
    local DEBUG_OUT="$DEBUG_DIR/$(basename ${CONFIG%.*})"

    echo ""
    echo "=========================================="
    echo "[$EXP_ID] 开始推理"
    echo "Config: $CONFIG"
    echo "Checkpoint: $CHECKPOINT"
    echo "Vis Dir: $SAVE_DIR"
    echo "Debug Dir: $DEBUG_OUT"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU python inference_detailed.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$VIS_DIR" \
        --limit $LIMIT \
        $( $PROGRESS && echo "--progress" ) \
        $( $SAVE_DEBUG && echo "--save-intermediate-dir $DEBUG_DIR" ) \
        --save-attn "$SAVE_ATTN"

    echo "[$EXP_ID] 推理完成!"
    echo "可视化: $SAVE_DIR"
    if $SAVE_DEBUG; then
        echo "中间结果: $DEBUG_OUT"
    fi
}

# 需要运行哪些就在下方取消注释
# doit "A1" "$CONFIG_A1" "$CKPT_A1"
# doit "A1NC" "$CONFIG_A1NC" "$CKPT_A1NC"
#doit "A2" "$CONFIG_A2" "$CKPT_A2"
# doit "B2" "$CONFIG_B2" "$CKPT_B2"
#doit "C2" "$CONFIG_C2" "$CKPT_C2"
#doit "D2" "$CONFIG_D2" "$CKPT_D2"
#doit "E1" "$CONFIG_E1" "$CKPT_E1"
#doit "E2" "$CONFIG_E2" "$CKPT_E2"
# doit "F1a" "$CONFIG_F1a" "$CKPT_F1a"
# doit "F1b" "$CONFIG_F1b" "$CKPT_F1b"
# doit "F2a" "$CONFIG_F2a" "$CKPT_F2a"
# doit "F2b" "$CONFIG_F2b" "$CKPT_F2b"
# doit "F2c" "$CONFIG_F2c" "$CKPT_F2c"
# doit "F2d" "$CONFIG_F2d" "$CKPT_F2d"
doit "F2d1" "$CONFIG_F2d1" "$CKPT_F2d1"
doit "F2d2" "$CONFIG_F2d2" "$CKPT_F2d2"
doit "F2d3" "$CONFIG_F2d3" "$CKPT_F2d3"
# doit "F2d-xsam" "$CONFIG_F2dXSAM" "$CKPT_F2dXSAM"

echo "=========================================="
echo "所有推理完成!"
echo "可视化查看: $VIS_DIR"
echo "中间结果查看: $DEBUG_DIR"
echo "=========================================="