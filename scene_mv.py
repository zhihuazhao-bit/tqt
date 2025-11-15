import os
import json
import shutil
from pathlib import Path

# 加载 english_scene_dict.json 文件
with open('/root/tqdm/dataset/road3d/english_scene_dict-foggy+dusk.json', 'r') as f:
    scene_dict = json.load(f)

# 定义输入目录和输出目录
base_dir = '/root/tqdm/dataset'
output_dir = '/root/tqdm/dataset/collected_images'
output_dir = os.path.join(output_dir, base_dir.split('/')[-1])
os.makedirs(output_dir, exist_ok=True)

# 遍历 training, validation, testing 目录
for split in [
    # 'training', 'validation', 'testing'
    'orfdv2'
    ]:
    split_dir = os.path.join(base_dir, split)
    if not os.path.exists(split_dir):
        continue

    # 遍历每个场景
    for scene_name in os.listdir(split_dir):
        scene_path = os.path.join(split_dir, scene_name)
        image_data_dir = os.path.join(scene_path, 'image_data')

        # 检查 image_data 文件夹是否存在
        if not os.path.isdir(image_data_dir):
            print(f"Skipping {scene_path}, 'image_data' not found.")
            continue

        # 获取 image_data 文件夹中的第一张图片
        image_files = sorted(Path(image_data_dir).glob('*.*'))  # 匹配所有文件
        if not image_files:
            print(f"No images found in {image_data_dir}.")
            continue

        first_image = image_files[0]

        # 从 JSON 文件中获取 weather, light, road 信息
        scene_info = scene_dict.get(scene_name[1:].replace('_', '-'), {})
        weather = scene_info.get('weather', 'unknown')
        light = scene_info.get('light', 'unknown')
        road = scene_info.get('road', 'unknown')

        # 构造输出文件名
        output_filename = f"{weather}_{light}_{road}_{scene_name}{first_image.suffix}"
        output_path = os.path.join(output_dir, output_filename)

        # 复制图片到目标目录
        shutil.copy(first_image, output_path)
        print(f"Copied {first_image} to {output_path}")