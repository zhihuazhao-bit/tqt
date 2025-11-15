import torch
weight = torch.load('/root/tqdm/weight/pretrained/EVA02_CLIP_B_psz16_s8B.pt', map_location='cpu')
weight2 = torch.load('/root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-r2r-512-light-traversable-pixel-proj-cls-prefix/best_mIoU_iter_2000.pth', map_location='cpu')
print(weight.keys())