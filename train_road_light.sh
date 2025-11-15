# python train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix.py
GPUS=4
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --launcher pytorch --gpus $GPUS

python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth --launcher pytorch 
# >./work_dirs/test_logs/road-test_b5k_light_traversable-pixel-proj-cls-prefix-fix-fliperror.log 2>&1 &

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth --launcher none

# python interference.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth --launcher none --show-dir ./work_dirs/test/road-tqt-eva-b-light-traversable-pixel-proj-cls-prefix-slide-flip