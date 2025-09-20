#!/bin/bash

for size in l b; do
    for i in terrian terrian-prefixRegion; do
        nohup python test.py \
            --config configs/tqdm/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-sun-${i}.py \
            --checkpoint work_dirs/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-sun-${i}/iter_5000.pth \
            --launcher none \
            --show \
            > work_dirs/logs/test_${size}_${i}.log 2>&1 &
        # 等待当前任务完成再执行下一个（可选）
        wait
    done
done