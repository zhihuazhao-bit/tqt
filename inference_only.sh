#!/bin/bash
# 消融实验推理脚本 - 仅保存推理结果（可视化图像和指标）
# 不保存中间结果（score map / OT pi / 注意力权重等）
#
# 使用方法:
#   1. 填写各实验的 CONFIG 与 CKPT 路径
#   2. 设置输出目录
#   3. 运行: bash inference_only.sh

GPU=0

# 可视化输出根目录（按 config 名称自动分子目录）
VIS_DIR="./work_dirs/inference_results"

# 最大推理样本数 (-1 表示全部)
LIMIT=-1
# 是否显示进度条
PROGRESS=true
# 可视化缩放比例
VIS_SCALE=1.0

# ============================================================================
# Unknown 场景配置 (来自 utils/scene_config.py)
# ============================================================================
SCENES_ORFD="0609-1923,2021-0223-1756"
SCENES_ROAD3D="2021-0403-1744,0602-1107,2021-0223-1857,2021-0403-1736"
SCENES_ORFD2ROAD="0609-1924,0609-1923,2021-0403-1736,2021-0223-1857"

# 根据数据集名称获取场景列表
get_scenes() {
    local DATASET=$1
    case "$DATASET" in
        orfd)
            echo "$SCENES_ORFD"
            ;;
        road3d|road)
            echo "$SCENES_ROAD3D"
            ;;
        orfd2road)
            echo "$SCENES_ORFD2ROAD"
            ;;
        *)
            echo ""  # 留空则自动检测
            ;;
    esac
}

# ============================================================================
# 在此填入各实验的 config / checkpoint
# ============================================================================

# Ours (ORFD)
CKPT_OURS_ORFD="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/20251219_1354/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/best_mIoU_iter_2000.pth"
CONFIG_OURS_ORFD="configs/ablation/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau.py"

# Ours (Road3D)
CKPT_OURS_ROAD="/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1320/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/best_mIoU_iter_4000.pth"
CONFIG_OURS_ROAD="configs/ablation_road/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau.py"

# LearnableOnly-M2F-NoPretrain
# CKPT_BASELINE="/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain/20251225_1726/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain/best_mIoU_iter_4000.pth"
# CONFIG_BASELINE="configs/ablation/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain.py"

# ============================================================================

echo "=========================================="
echo "推理脚本 - 仅保存可视化结果和指标"
echo "GPU: $GPU"
echo "输出目录: $VIS_DIR"
echo "=========================================="

# 推理函数
# 参数: EXP_ID CONFIG CHECKPOINT DATASET [CUSTOM_SCENE_LIST]
# DATASET: orfd, road3d, orfd2road (用于自动选择场景)
# CUSTOM_SCENE_LIST: 可选，手动指定场景列表（覆盖自动选择）
infer() {
    local EXP_ID=$1
    local CONFIG=$2
    local CHECKPOINT=$3
    local DATASET=$4
    local CUSTOM_SCENES=$5

    # 根据数据集自动选择场景，或使用自定义场景
    local SCENE_LIST
    if [ -n "$CUSTOM_SCENES" ]; then
        SCENE_LIST="$CUSTOM_SCENES"
    else
        SCENE_LIST=$(get_scenes "$DATASET")
    fi

    if [ ! -f "$CHECKPOINT" ]; then
        echo "警告: Checkpoint 文件不存在: $CHECKPOINT, 跳过 $EXP_ID"
        return 1
    fi

    local SAVE_DIR="$VIS_DIR/$(basename ${CONFIG%.*})"

    echo ""
    echo "=========================================="
    echo "[$EXP_ID] 开始推理"
    echo "Config: $CONFIG"
    echo "Checkpoint: $CHECKPOINT"
    echo "Dataset: $DATASET"
    echo "Scene List: ${SCENE_LIST:-auto}"
    echo "Output: $SAVE_DIR"
    echo "=========================================="

    CUDA_VISIBLE_DEVICES=$GPU python inference_detailed.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$VIS_DIR" \
        --limit $LIMIT \
        --vis-scale $VIS_SCALE \
        --save-attn False \
        --save-intermediate-dir None \
        $( $PROGRESS && echo "--progress" ) \
        $( [ -n "$SCENE_LIST" ] && echo "--force-scene-list $SCENE_LIST" )

    echo "[$EXP_ID] 推理完成!"
    echo "结果保存在: $SAVE_DIR"
}

# ============================================================================
# 取消注释需要运行的实验
# 格式: infer "实验ID" "CONFIG" "CHECKPOINT" "数据集类型"
# 数据集类型: orfd, road3d, orfd2road
# ============================================================================

# ORFD 数据集实验
# infer "Ours-ORFD" "$CONFIG_OURS_ORFD" "$CKPT_OURS_ORFD" "orfd"

# Road3D 数据集实验
# infer "Ours-Road3D" "$CONFIG_OURS_ROAD" "$CKPT_OURS_ROAD" "road3d"

# ORFD2Road 跨域实验
infer "Ours-ORFD2Road" "$CONFIG_OURS_ROAD" "$CKPT_OURS_ORFD" "orfd2road"

# 手动指定场景（第5个参数覆盖自动选择）
# infer "Custom" "$CONFIG_OURS_ORFD" "$CKPT_OURS_ORFD" "orfd" "0609-1923"

echo ""
echo "=========================================="
echo "所有推理完成!"
echo "结果查看: $VIS_DIR"
echo "=========================================="
