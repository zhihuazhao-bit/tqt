import os
import cv2
import numpy as np
from tqdm import tqdm

def convert_fillcolor_to_labeltrainids(gt_image_dir, output_dir=None):
    """
    将 _fillcolor.png 图片转换为 _labelTrainIds.png 格式
    
    Args:
        gt_image_dir: 包含 _fillcolor.png 文件的目录路径
        output_dir: 输出目录，如果为 None 则输出到同一目录
    """
    if output_dir is None:
        output_dir = gt_image_dir
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有 _fillcolor.png 文件
    fillcolor_files = [f for f in os.listdir(gt_image_dir) if f.endswith('_fillcolor.png')]
    
    if not fillcolor_files:
        print(f"No fillcolor images found in {gt_image_dir}")
        return
    
    print(f"Found {len(fillcolor_files)} fillcolor images to convert in {gt_image_dir}")
    
    for filename in tqdm(fillcolor_files, desc="Converting labels"):
        # 构建输入和输出路径
        input_path = os.path.join(gt_image_dir, filename)
        
        # 生成输出文件名：将 _fillcolor.png 替换为 _labelTrainIds.png
        output_filename = filename.replace('_fillcolor.png', '_labelTrainIds.png')
        output_path = os.path.join(output_dir, output_filename)
        
        # 读取图片
        try:
            label_image = cv2.imread(input_path)
            if label_image is None:
                print(f"Warning: Could not read {input_path}")
                continue
                
            # 转换颜色空间
            label_image = cv2.cvtColor(label_image, cv2.COLOR_BGR2RGB)
            
            # 获取图片尺寸
            height, width = label_image.shape[:2]
            
            # 创建标签数组
            label = np.zeros((height, width), dtype=np.uint8)
            
            # 根据蓝色通道值设置标签 (R通道值 > 200 的像素设为1)
            label[label_image[:,:,2] > 200] = 1
            
            # 保存标签图片
            cv2.imwrite(output_path, label)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

def process_all_scenes(base_dir):
    """
    遍历 training、testing、val 下面的每个场景子文件夹
    
    Args:
        base_dir: ORFD 数据集的根目录
    """
    splits = ['training', 'testing', 'validation']
    
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        
        if not os.path.exists(split_dir):
            print(f"Warning: {split_dir} does not exist, skipping...")
            continue
            
        print(f"\nProcessing {split} split...")
        
        # 遍历该split下的所有场景文件夹
        scene_folders = [f for f in os.listdir(split_dir) 
                        if os.path.isdir(os.path.join(split_dir, f))]
        
        if not scene_folders:
            print(f"No scene folders found in {split_dir}")
            continue
            
        print(f"Found {len(scene_folders)} scene folders in {split}")
        
        for scene_folder in tqdm(scene_folders):
            scene_path = os.path.join(split_dir, scene_folder)
            gt_image_dir = os.path.join(scene_path, 'gt_image')
            
            if os.path.exists(gt_image_dir):
                print(f"\nProcessing scene: {scene_folder}")
                convert_fillcolor_to_labeltrainids(gt_image_dir)
            else:
                print(f"Warning: gt_image directory not found in {scene_path}")

# 使用示例
if __name__ == "__main__":
    # ORFD 数据集根目录
    base_dir = "/root/tqdm/datasets/ORFD"
    
    # 处理所有场景
    process_all_scenes(base_dir)
    
    print("\nAll conversions completed!")