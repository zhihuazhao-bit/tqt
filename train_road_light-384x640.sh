# python train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-light-traversable-pixel-proj-cls-prefix.py
GPUS=4
# python3 -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640.py --launcher pytorch --gpus $GPUS

python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_5000.pth --launcher pytorch 
# --show --show-dir ./work_dirs/test/road-tqt-eva-b-light-traversable-pixel-proj-cls-prefix-384x640

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_2000.pth --launcher none

# python interference.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640.py  --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_5000.pth --launcher none --show-dir ./work_dirs/test/road-tqt-eva-b-light-traversable-pixel-proj-cls-prefix-384x640