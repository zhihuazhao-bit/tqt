#!/bin/bash
# 单实验测试入口（默认用于 F2c Patch-FPN 消融）
# 修改 CKPT 路径后直接运行: bash test.sh

GPU=0
CONFIG="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_noprompt.py"
CKPT="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_noprompt/PLACEHOLDER_TIMESTAMP/exp_224_eva02_sneotTrue_patchfpn_noprompt/best_mIoU_iter_1000.pth"  # TODO: 填写训练完成的 checkpoint

SAVE_DIR=$(dirname "$CKPT")/test_results_patchfpn/

echo "=========================================="
echo "[Test] Patch-FPN Ablation (F2c)"
echo "GPU: $GPU"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Save Dir: $SAVE_DIR"
echo "=========================================="

if [ ! -f "$CKPT" ]; then
	echo "警告: 未找到 checkpoint: $CKPT"
	echo "请先更新 CKPT 路径后再运行本脚本."
	exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU python test.py \
	--config $CONFIG \
	--checkpoint $CKPT \
	--eval mIoU mFscore \
	--show-dir $SAVE_DIR \
	--save_dir $SAVE_DIR

echo "=========================================="
echo "[Test] 完成"
echo "CSV 保存位置: $SAVE_DIR"
echo "=========================================="