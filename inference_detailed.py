import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset

import models  # 导入自定义模型注册


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Detailed inference with TP/FP/FN visualization and metrics.')
    parser.add_argument('--config', default="/root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj.py", help='Path to the mmseg config file.')
    parser.add_argument('--checkpoint', default="/root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj/best_mIoU_iter_1000.pth", help='Model checkpoint for inference.')
    parser.add_argument('--output-dir', default='./work_dirs/detailed_inference', help='Directory to save detailed results.')
    parser.add_argument('--device', default='cuda:0', help='Device for inference, e.g. "cuda:0" or "cpu".')
    parser.add_argument('--limit', type=int, default=-1, help='Maximum number of frames to process (-1 for all).')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar during processing.')
    parser.add_argument('--traversable-class', type=int, default=1, help='Class index for traversable regions. If None, auto-detect from class names.')
    parser.add_argument('--vis-scale', type=float, default=2, help='Scale factor for output visualization (default: 0.5).')
    parser.add_argument('--show-dir', default=None, help='Directory to save visualizations (overrides output-dir).')
    return parser.parse_args()


def _patch_dataset_cfg(cfg: Dict, args: argparse.Namespace) -> Dict:
    """不再需要 patch dataset，直接使用 config 中的配置"""
    return cfg


def _to_display_uint8(img: np.ndarray, target_shape: Optional[tuple] = None) -> np.ndarray:
    """Convert image to uint8 RGB format, optionally resize."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if target_shape is not None and img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return img


def _normal_to_rgb(normal: Optional[np.ndarray], target_shape: tuple) -> np.ndarray:
    """Convert normal map to RGB uint8 visualization."""
    if normal is None:
        return np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
    normal = normal.astype(np.float32)
    vmin, vmax = normal.min(), normal.max()
    if vmax - vmin < 1e-6:
        normal[:] = 0
    else:
        normal = (normal - vmin) / (vmax - vmin)
    normal = (normal * 255).clip(0, 255).astype(np.uint8)
    return _to_display_uint8(normal, target_shape)


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Compute precision, recall, F1, accuracy from confusion matrix."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    }


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    # 导入 SimpleTokenizer (参考 legacy 代码)
    from models.backbones.utils import SimpleTokenizer

    # 初始化 SimpleTokenizer
    tokenizer = SimpleTokenizer()
    tokens = np.zeros((len(cfg.class_names)), dtype=np.int64)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    
    # 对 class_names 中的每个类别名称进行分词
    for i, class_name in enumerate(cfg.class_names):
        token = [sot_token] + tokenizer.encode(class_name) + [eot_token]
        tokens[i] = len(token) + 12
    cfg.model.context_length = int(tokens.max())
    cfg.model.eva_clip.context_length = int(tokens.max()) + 8

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    dataset = build_dataset(cfg.data.test)
    workers = cfg.data.get('workers_per_gpu', 0)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False
    )

    # 获取 data_root 用于构建完整路径
    data_root = dataset.data_root if hasattr(dataset, 'data_root') else None
    img_dir = dataset.img_dir if hasattr(dataset, 'img_dir') else None
    ann_dir = dataset.ann_dir if hasattr(dataset, 'ann_dir') else None

    device = torch.device(args.device)
    cfg.model.class_names = list(dataset.CLASSES)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    
    if args.checkpoint == 'None':
        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE
    elif "CLIP-ViT" in args.checkpoint:
        model.backbone.init_weights(args.checkpoint)
        model.text_encoder.init_weights(args.checkpoint)
        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE
    else:
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            print('"CLASSES" not found in meta, use dataset.CLASSES instead')
            model.CLASSES = dataset.CLASSES
        
        if 'PALETTE' in checkpoint.get('meta', {}):
            model.PALETTE = checkpoint['meta']['PALETTE']
        else:
            print('"PALETTE" not found in meta, use dataset.PALETTE instead')
            model.PALETTE = dataset.PALETTE

    # 用 MMDataParallel 包装模型（关键！）
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # 自动检测可通行类别索引
    if args.traversable_class is None:
        # 尝试从类别名称中自动检测
        class_names = [str(c).lower() for c in model.module.CLASSES]
        traversable_keywords = ['traversable', 'vehicle-accessible', 'road', 'drivable']
        
        traversable_class = 0  # 默认
        for idx, name in enumerate(class_names):
            if any(keyword in name for keyword in traversable_keywords):
                traversable_class = idx
                break
        
        print(f'Auto-detected traversable class: {traversable_class} ({model.module.CLASSES[traversable_class]})')
    else:
        traversable_class = args.traversable_class
        print(f'Using specified traversable class: {traversable_class} ({model.module.CLASSES[traversable_class]})')

    # 确定输出目录
    if args.show_dir is not None:
        output_root = Path(args.show_dir)
    else:
        config_name = os.path.basename(args.config).split('.')[0]
        output_root = Path(args.output_dir) / config_name
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    if args.limit >= 0:
        total = min(total, args.limit)
    progress_bar = mmcv.ProgressBar(total) if args.progress else None

    # Global confusion matrix
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    processed = 0

    for data in data_loader:
        if 0 <= args.limit <= processed:
            break

        # 直接传递 data，就像 test.py 中一样
        with torch.no_grad():
            result = model(return_loss=False, **data)

        # 参考 test.py 的方式提取 img_metas
        img_metas = data['img_metas'][0].data[0]
        img_meta = img_metas[0]  # batch size = 1

        # Get prediction and convert to binary
        seg_pred = result[0].astype(np.uint8)
        pred_binary = (seg_pred == traversable_class).astype(np.uint8)

        # 构建完整的图像路径
        # img_meta['filename'] 通常是相对路径，需要和 img_prefix 或 data_root 组合
        filename = img_meta.get('ori_filename', img_meta.get('filename'))
        
        # 如果 filename 已经是绝对路径，直接使用；否则和 img_prefix 组合
        if os.path.isabs(filename):
            rgb_path = filename
        else:
            # 尝试从 img_meta 中获取 img_prefix，或使用 dataset 的配置
            img_prefix = img_meta.get('img_prefix', img_dir)
            if img_prefix:
                rgb_path = os.path.join(img_prefix, filename)
            else:
                rgb_path = filename
        
        # 从完整路径中提取场景名称和文件名
        path_parts = Path(rgb_path).parts
        base_name = Path(rgb_path).stem
        
        # 查找 'testing' 或 'training' 或 'validation' 后的场景目录
        scene_name = 'unknown'
        for i, part in enumerate(path_parts):
            if part in ['testing', 'training', 'validation'] and i + 1 < len(path_parts):
                scene_name = path_parts[i + 1]
                break

        rgb_image_ori = mmcv.imread(str(rgb_path), channel_order='rgb')
        oriHeight, oriWidth = rgb_image_ori.shape[:2]

        # Resize prediction to original size
        pred_resized = cv2.resize(pred_binary, (oriWidth, oriHeight), interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # Load normal map
        # 从 img_meta 获取 sne_filename，或从 rgb_path 推断
        sne_filename = img_meta.get('sne_filename')
        if sne_filename:
            # 如果 sne_filename 已经在 img_meta 中（由 pipeline 设置）
            if os.path.isabs(sne_filename):
                normal_path = sne_filename
            else:
                # 相对路径，需要和某个 prefix 组合
                sne_prefix = img_meta.get('seg_prefix', ann_dir)
                if sne_prefix:
                    # sne 通常和 image 在同级目录
                    normal_path = str(rgb_path).replace('image_data', 'surface_normal_d2net_v3')
                else:
                    normal_path = sne_filename
        else:
            # 从 rgb_path 推断 normal map 路径
            normal_path = str(rgb_path).replace('image_data', 'surface_normal_d2net_v3')
        
        normal_path = Path(normal_path)
        if normal_path.exists():
            sn_image_ori = mmcv.imread(str(normal_path), flag='unchanged')
            sn_image_ori = _normal_to_rgb(sn_image_ori, (oriHeight, oriWidth))
        else:
            sn_image_ori = np.zeros((oriHeight, oriWidth, 3), dtype=np.uint8)

        # Load ground truth
        gt_tensor = data.get('gt_semantic_seg')
        if gt_tensor is not None:
            gt_mask = gt_tensor.data[0].squeeze().cpu().numpy().astype(np.uint8)
            gt_binary = (gt_mask == traversable_class).astype(np.uint8)
        else:
            # 从 rgb_path 推断 label 路径
            label_path = str(rgb_path).replace('image_data', 'gt_image').replace('.png', '_labelTrainIds.png')
            label_path = Path(label_path)
            if label_path.exists():
                label_img = mmcv.imread(str(label_path), flag='unchanged')
                if label_img.ndim == 2:
                    gt_binary = (label_img == traversable_class).astype(np.uint8)
                else:
                    # Assume colored label, convert to grayscale and threshold
                    gt_gray = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)
                    gt_binary = (gt_gray > 128).astype(np.uint8)
            else:
                gt_binary = np.zeros((oriHeight, oriWidth), dtype=np.uint8)

        # Compute confusion matrix
        tp = np.sum((pred_resized == 1) & (gt_binary == 1))
        fp = np.sum((pred_resized == 1) & (gt_binary == 0))
        fn = np.sum((pred_resized == 0) & (gt_binary == 1))
        tn = np.sum((pred_resized == 0) & (gt_binary == 0))

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn

        # Build masks for visualization overlays
        tp_mask = (pred_resized == 1) & (gt_binary == 1)
        fp_mask = (pred_resized == 1) & (gt_binary == 0)
        fn_mask = (pred_resized == 0) & (gt_binary == 1)

        # Overlay color-coded results with transparency for easier inspection
        alpha = 0.5
        rgb_float = rgb_image_ori.astype(np.float32)
        overlay = np.zeros_like(rgb_float)
        overlay[tp_mask] = [0, 255, 0]
        overlay[fp_mask] = [255, 0, 0]
        overlay[fn_mask] = [0, 0, 255]
        any_mask = tp_mask | fp_mask | fn_mask
        rgb_float[any_mask] = (
            (1 - alpha) * rgb_float[any_mask] + alpha * overlay[any_mask]
        )
        vis_image = rgb_float.astype(np.uint8)

        # GT visualization (yellow for traversable)
        gt_vis = rgb_image_ori.copy()
        gt_vis[gt_binary == 1] = [255, 255, 0]

        # Concatenate images: [RGB | Normal]
        #                      [Overlay | GT]
        img_cat1 = np.concatenate((rgb_image_ori, sn_image_ori), axis=1)
        img_cat2 = np.concatenate((vis_image, gt_vis), axis=1)
        img_cat = np.concatenate((img_cat1, img_cat2), axis=0)

        # Compute metrics for the current frame and render a text banner above the visualization
        metrics = compute_metrics(tp, fp, fn, tn)
        text = f"P:{metrics['precision']:.3f} R:{metrics['recall']:.3f} F1:{metrics['f1']:.3f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_band_height = max(text_size[1] + baseline + 20, 60)
        text_canvas = np.zeros((text_band_height, img_cat.shape[1], 3), dtype=np.uint8)
        text_x = (img_cat.shape[1] - text_size[0]) // 2
        text_y = (text_band_height + text_size[1]) // 2
        cv2.putText(text_canvas, text, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)
        img_cat = np.concatenate((text_canvas, img_cat), axis=0)

        # Optional resizing for saving; upscale uses bicubic for smoother results
        if args.vis_scale != 1.0:
            scale = args.vis_scale
            new_w = int(img_cat.shape[1] * scale)
            new_h = int(img_cat.shape[0] * scale)
            interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
            img_cat = cv2.resize(img_cat, (new_w, new_h), interpolation=interpolation)

        # Save visualization (convert RGB to BGR for cv2)
        img_cat_bgr = cv2.cvtColor(img_cat, cv2.COLOR_RGB2BGR)
        save_dir = os.path.join(output_root)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'detailed_{scene_name}_{base_name}.png')
        cv2.imwrite(str(save_path), img_cat_bgr)

        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.bar.finish()

    # Compute and print overall metrics
    overall_metrics = compute_metrics(total_tp, total_fp, total_fn, total_tn)
    
    print(f'\n{"="*60}')
    print(f'Processed {processed} images')
    print(f'{"="*60}')
    print(f'Overall Metrics:')
    print(f'  Precision: {overall_metrics["precision"]:.4f}')
    print(f'  Recall:    {overall_metrics["recall"]:.4f}')
    print(f'  F1 Score:  {overall_metrics["f1"]:.4f}')
    print(f'  Accuracy:  {overall_metrics["accuracy"]:.4f}')
    print(f'{"="*60}')
    print(f'Confusion Matrix:')
    print(f'  TP: {total_tp:>10d}  |  FP: {total_fp:>10d}')
    print(f'  FN: {total_fn:>10d}  |  TN: {total_tn:>10d}')
    print(f'{"="*60}')
    
    # Save metrics to file
    metrics_file = output_root / 'metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write(f'Processed images: {processed}\n')
        f.write(f'\nOverall Metrics:\n')
        f.write(f'Precision: {overall_metrics["precision"]:.4f}\n')
        f.write(f'Recall:    {overall_metrics["recall"]:.4f}\n')
        f.write(f'F1 Score:  {overall_metrics["f1"]:.4f}\n')
        f.write(f'Accuracy:  {overall_metrics["accuracy"]:.4f}\n')
        f.write(f'\nConfusion Matrix:\n')
        f.write(f'TP: {total_tp}\n')
        f.write(f'FP: {total_fp}\n')
        f.write(f'FN: {total_fn}\n')
        f.write(f'TN: {total_tn}\n')

    print(f'\nResults saved to: {output_root}')
    print(f'Metrics saved to: {metrics_file}')


if __name__ == '__main__':
    main()
