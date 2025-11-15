# python train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_10k-r2r-512-road-traversable-pixel-proj-cls-prefix.py
GPUS=4
python3 -m torch.distributed.launch --nproc_per_node=$GPUS train.py --config /root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-road-traversable-pixel-proj-cls-prefix.py --launcher pytorch --gpus $GPUS