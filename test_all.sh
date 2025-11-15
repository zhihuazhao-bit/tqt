# python train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-weather-traversable-pixel-proj-cls-prefix.py
GPUS=4
scene_type='all'

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-add.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-add/best_mIoU_iter_2000.pth --launcher pytorch
wait

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nocrossattn.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nocrossattn/best_mIoU_iter_5000.pth --launcher pytorch
# wait

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nopromptcls.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nopromptcls/best_mIoU_iter_2000.pth --launcher pytorch
# wait

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-useeva.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-useeva/best_mIoU_iter_5000.pth --launcher pytorch
wait

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-concat.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-concat/iter_1000.pth --launcher pytorch

python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-final-bk/iter_1000.pth --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224/best_mIoU_iter_5000.pth --launcher pytorch

# python -m torch.distributed.launch --nproc_per_node=$GPUS  test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-$scene_type-traversable-pixel-proj-cls-prefix-224x224/best_mIoU_iter_2000.pth --launcher pytorch

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_2000.pth

# python test.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_2000.pth --launcher none

# python interference.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_2000.pth --launcher none --show-dir ./work_dirs/test/road-tqt-eva-b-$scene_type-traversable-pixel-proj-cls-prefix-384x640
# python interference.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640.py --checkpoint /root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-$scene_type-traversable-pixel-proj-cls-prefix-384x640/best_mIoU_iter_2000.pth --scene-dir /root/tqdm/dataset/road3d/testing/y2021_0403_1744 --progress