#!/bin/bash

for size in b; do
    for i in sufficient; do
        nohup python test.py \
            --config configs/tqdm/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-${i}-terrian.py \
            --checkpoint work_dirs/weights/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-${i}-terrian-fix-ema/best_mIoU_iter_4000.pth \
            --launcher none \
            --show \
            > work_dirs/test_logs/test_${size}_${i}_terrian-fix-ema.log 2>&1 &
        # 等待当前任务完成再执行下一个（可选）
        wait
    done
done