import os
import shutil
from tqdm import tqdm

# 路径配置
base_dir = '/root/tqdm/datasets/gta/raw'
image_dir = os.path.join(base_dir, 'images')
label_dir = os.path.join(base_dir, 'labels')

split_files = {
    'train': '/root/tqdm/datasets/gta/gtav_split_train.txt',
    'valid': '/root/tqdm/datasets/gta/gtav_split_val.txt',
    'test': '/root/tqdm/datasets/gta/gtav_split_test.txt'
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

for split, split_file in split_files.items():
    # 创建目标文件夹
    img_target = os.path.join(image_dir, split)
    lbl_target = os.path.join(label_dir, split)
    ensure_dir(img_target)
    ensure_dir(lbl_target)

    # 读取图片名
    with open(split_file, 'r') as f:
        names = [line.strip() for line in f if line.strip()]

    for name in tqdm(names):
        # 复制图片
        src_img = os.path.join(image_dir, name)
        dst_img = os.path.join(img_target, name)
        if not os.path.exists(dst_img):
            if os.path.exists(src_img):
                shutil.move(src_img, dst_img)
            else:
                print(f'Image not found: {src_img}')

            # 复制标签
            src_lbl = os.path.join(label_dir, name)
            dst_lbl = os.path.join(lbl_target, name)
            if os.path.exists(src_lbl):
                shutil.move(src_lbl, dst_lbl)
            else:
                print(f'Label not found: {src_lbl}')
        else:
            print(f'Already exists: {dst_img}')