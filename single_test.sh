#!/bin/bash

# for size in b; do
#     for i in light road weather; do
#         nohup python test.py \
#             --config configs/tqdm/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-${i}-traversable-pixel-proj-cls-prefix.py \
#             --checkpoint work_dirs/weights/tqdm_eva_vit-${size}_1e-5_5k-o2o-512-${i}-traversable-pixel-proj-cls-prefix/best_mIoU_iter_4000.pth \
#             --launcher none \
#             --show \
#             > work_dirs/test_logs/test_${size}_${i}_traversable-pixel-proj-cls-prefix.log 2>&1 &
#         # 等待当前任务完成再执行下一个（可选）
#         wait
#     done
# done
# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/weights/tqt_eva_vit-b_1e-5_5k-o2o-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_3000.pth --launcher none --show

# nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/weights/tqt_eva_vit-b_1e-5_5k-o2o-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_3000.pth --launcher none --show >./work_dirs/test_logs/test_b_light_traversable-pixel-proj-cls-prefix.log 2>&1 &

# wait

# nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-road-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-road-traversable-pixel-proj-cls-prefix/best_mIoU_iter_5000.pth --launcher none --show >./work_dirs/test_logs/test_b_road_traversable-pixel-proj-cls-prefix.log 2>&1 &

# wait

# nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-weather-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-weather-traversable-pixel-proj-cls-prefix/best_mIoU_iter_4000.pth --launcher none --show >./work_dirs/test_logs/test_b_weather_traversable-pixel-proj-cls-prefix.log 2>&1 &

# wait

# nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_light_traversable-pixel-proj-cls-prefix.log 2>&1 &
# wait

# nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-road-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-road-traversable-pixel-proj-cls-prefix/best_mIoU_iter_8000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_road_traversable-pixel-proj-cls-prefix.log 2>&1 &
# wait

nohup python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-weather-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-weather-traversable-pixel-proj-cls-prefix/best_mIoU_iter_3000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_weather_traversable-pixel-proj-cls-prefix.log 2>&1 &
wait

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_light_traversable-pixel-proj-cls-prefix.log
# wait

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-road-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-road-traversable-pixel-proj-cls-prefix/best_mIoU_iter_8000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_road_traversable-pixel-proj-cls-prefix.log
# wait

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-weather-traversable-pixel-proj-cls-prefix.py --checkpoint work_dirs/tqt_eva_vit-b_1e-5_10k-r2r-512-weather-traversable-pixel-proj-cls-prefix/best_mIoU_iter_3000.pth --launcher none --show >./work_dirs/test_logs/road-test_b_weather_traversable-pixel-proj-cls-prefix.log
# wait