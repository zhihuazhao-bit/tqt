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
from utils.scene_config import DATASET_UNKNOWN_SCENES, ABNORMAL_SCENES


def detect_dataset_from_config(config_path: str) -> str:
    """根据 config 路径自动检测数据集类型"""
    config_lower = config_path.lower()
    if 'road3d' in config_lower or 'road2road' in config_lower:
        return 'road3d'
    elif 'orfd' in config_lower:
        return 'orfd'
    return 'orfd'  # 默认 orfd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Detailed inference with TP/FP/FN visualization and metrics.')
    parser.add_argument('--config', default="configs/ablation/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_L2.py", help='Path to the mmseg config file.')
    parser.add_argument('--checkpoint', default="/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-l2/20251212_2322/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_l2/best_mIoU_iter_1000.pth", help='Model checkpoint for inference.')
    parser.add_argument('--output-dir', default='./work_dirs/detailed_inference', help='Directory to save detailed results.')
    parser.add_argument('--device', default='cuda:0', help='Device for inference, e.g. "cuda:0" or "cpu".')
    parser.add_argument('--limit', type=int, default=-1, help='Maximum number of frames to process (-1 for all).')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar during processing.')
    parser.add_argument('--traversable-class', type=int, default=1, help='Class index for traversable regions. If None, auto-detect from class names.')
    parser.add_argument('--vis-scale', type=float, default=2, help='Scale factor for output visualization (default: 0.5).')
    parser.add_argument('--show-dir', default=None, help='Directory to save visualizations (overrides output-dir).')
    parser.add_argument('--save-intermediate-dir', default='./attn', help='Directory to save intermediate tensors (score maps / OT pi).')
    parser.add_argument('--save-attn', default=True, help='Save attention weight matrices returned by the model.')
    parser.add_argument('--force-scene-list', default=None, help="Comma-separated scene ids to restrict inference. If not specified, auto-detect unknown scenes based on dataset.")
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

    # 自动检测数据集并设置 unknown 场景列表
    if args.force_scene_list is None:
        dataset_name = detect_dataset_from_config(args.config)
        unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])
        args.force_scene_list = ','.join(unknown_scenes)
        print(f"[Auto-detect] Dataset: {dataset_name}, Unknown scenes: {unknown_scenes}")

    # 仅在推理时指定场景子集，不影响训练/验证配置
    if args.force_scene_list:
        if not hasattr(cfg, 'data') or not hasattr(cfg.data, 'test'):
            raise ValueError('config missing data.test, cannot set force_scene_list')
        cfg.data.test.force_scene_list = args.force_scene_list

    def _enable_return_attn(cfg_obj):
        """开启注意力返回：调整 test_cfg 与 decode_head 关键开关，匹配配置文件里的 return_attn 逻辑。"""
        cfg_obj.model.test_cfg = cfg_obj.model.get('test_cfg', {})
        cfg_obj.model.test_cfg['return_attn'] = True

        # 同步 decode_head 的相关开关/类型
        try:
            dec = cfg_obj.model.decode_head
            px = dec.pixel_decoder
            px['return_attn_weights'] = True
            # encoder 与 layer 类型切换到 Attn* 版本
            if 'encoder' in px:
                px.encoder['type'] = 'AttnDetrTransformerDecoder'
                if 'transformerlayers' in px.encoder:
                    px.encoder.transformerlayers['type'] = 'AttnDetrTransformerDecoderLayer'
                    # cross-attn 用 AttnMultiheadAttention
                    attn_cfgs = px.encoder.transformerlayers.get('attn_cfgs', None)
                    if isinstance(attn_cfgs, list) and len(attn_cfgs) >= 2:
                        # 第二个 usually cross-attn
                        attn_cfgs[1]['type'] = 'AttnMultiheadAttention'
        except Exception:
            # 若结构不同则忽略，保持安全
            pass

    # 若需要返回注意力，必须在构建模型前修改配置以创建对应模块
    if args.save_attn:
        _enable_return_attn(cfg)

    # 导入 SimpleTokenizer (参考 legacy 代码)
    from models.backbones.utils import SimpleTokenizer

    # 初始化 SimpleTokenizer
    tokenizer = SimpleTokenizer()
    tokens = np.zeros((len(cfg.class_names)), dtype=np.int64)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    
    # 对 class_names 中的每个类别名称进行分词，动态扩展文本上下文长度以容纳最长类别描述
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
        # 尝试从类别名称中自动检测，避免手动指定
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
    # 可选的进度条，方便长序列推理查看进度
    progress_bar = mmcv.ProgressBar(total) if args.progress else None

    # Global confusion matrix
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    processed = 0

    # 中间结果保存根目录
    debug_root = None
    if args.save_intermediate_dir is not None:
        config_name = os.path.basename(args.config).split('.')[0]
        debug_root = Path(args.save_intermediate_dir) / config_name
        debug_root.mkdir(parents=True, exist_ok=True)

    for data in data_loader:
        if 0 <= args.limit <= processed:
            break

        # 直接传递 data，就像 test.py 中一样
        with torch.no_grad():
            # 依照 mmseg test 流程，直接调用模型的 forward_test
            need_debug = args.save_intermediate_dir is not None
            result = model(
                return_loss=False,
                return_debug=need_debug,
                **data
            )

        # 参考 test.py 的方式提取 img_metas
        img_metas = data['img_metas'][0].data[0]
        img_meta = img_metas[0]  # batch size = 1

        # 解析模型输出（支持可选 attn/debug）
        debug_payload = None
        attn_matrix = None
        if args.save_attn and (args.save_intermediate_dir is not None):
            seg_pred, attn_matrix, debug_payload = result
        elif args.save_attn:
            seg_pred, attn_matrix = result
        elif args.save_intermediate_dir is not None:
            seg_pred, debug_payload = result
        else:
            seg_pred = result[0] if isinstance(result, (list, tuple)) else result

        # 统一转为 numpy HxW（simple_test 返回 list[np.ndarray]）
        if isinstance(seg_pred, list):
            seg_pred = seg_pred[0]
        if torch.is_tensor(seg_pred):
            seg_pred = seg_pred.detach().cpu().numpy()

        # Get prediction and convert to binary
        seg_pred = seg_pred.astype(np.uint8)
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
        # 按场景划分子目录
        img_cat_bgr = cv2.cvtColor(img_cat, cv2.COLOR_RGB2BGR)
        save_dir = os.path.join(output_root, scene_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'detailed_{base_name}.png')
        cv2.imwrite(str(save_path), img_cat_bgr)

        # 保存中间结果 (score map / pi / attn)
        if debug_root is not None:
            sample_root = debug_root / scene_name
            sample_root.mkdir(parents=True, exist_ok=True)

            # 若没有 debug_payload，则保持旧约定：不构建 concat，可选仅保存 attn npy
            if debug_payload is None:
                if args.save_attn and attn_matrix is not None:
                    attn_to_save = attn_matrix.detach().cpu() if torch.is_tensor(attn_matrix) else attn_matrix
                    torch.save(attn_to_save, str(sample_root / f'{base_name}_attn.pt'))
                # 直接跳过 concat 构建，保持原行为
                continue

            def _to_uint8_img(t: torch.Tensor) -> np.ndarray:
                t = t.detach().cpu()
                if t.ndim == 4:  # [B, C, H, W]
                    t = t[0]
                if t.ndim == 3 and t.shape[0] > 1:
                    if t.shape[0] >= 3:
                        t = t[:3]
                    else:
                        # t = t.mean(dim=0, keepdim=True)
                        t = t[1]
                if t.ndim == 3 and t.shape[0] == 1:
                    t = t.squeeze(0)
                t_min, t_max = t.min(), t.max()
                if float(t_max - t_min) < 1e-8:
                    arr = (t * 0).byte().numpy()
                else:
                    arr = ((t - t_min) / (t_max - t_min) * 255.0).clamp(0, 255).byte().numpy()
                if arr.ndim == 2:
                    arr = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                else:
                    arr = arr.transpose(1, 2, 0)
                return arr

            def _argmax_to_bw_trav(t: torch.Tensor, trav_idx: int) -> Optional[np.ndarray]:
                t = t.detach().cpu()
                if t.ndim == 4:
                    t = t[0]
                if t.ndim != 3:
                    return None
                label = t.argmax(dim=0).byte().numpy()
                mask = (label == trav_idx).astype(np.uint8) * 255
                return np.stack([mask] * 3, axis=-1)

            def _extract_attn(attn_raw):
                """提取注意力权重和可选的 spatial_shapes。

                返回 (tensor, spatial_shapes or None)。
                支持:
                  - tensor / np.ndarray
                  - list/tuple: 取最后一个元素递归
                  - dict: 支持格式
                        {'rgb': {'attn_weights': ..., 'spatial_shapes': ...}}
                        {'attn_weights': ..., 'spatial_shapes': ...}
                        {'attn': ...}
                        以及一般的键 'attn'/'attn_weights'/'weights'
                """
                if attn_raw is None:
                    return None, None
                # tensor
                if torch.is_tensor(attn_raw):
                    return attn_raw, None
                if isinstance(attn_raw, np.ndarray):
                    return torch.from_numpy(attn_raw), None
                # list/tuple: 取最后一个
                if isinstance(attn_raw, (list, tuple)) and len(attn_raw) > 0:
                    return _extract_attn(attn_raw[0])
                # dict: 优先解析 rgb/img 分支
                if isinstance(attn_raw, dict):
                    # rgb/img 子字典
                    for branch_key in ['rgb', 'img', 'image']:
                        if branch_key in attn_raw and isinstance(attn_raw[branch_key], dict):
                            sub = attn_raw[branch_key]
                            weights = sub.get('attn_weights', sub.get('attn', sub.get('weights')))
                            spatial_shapes = sub.get('spatial_shapes')
                            if weights is not None:
                                w_tensor, _ = _extract_attn(weights)
                                return w_tensor, spatial_shapes
                    # 直接取常用键
                    for key in ['attn_weights', 'attn', 'weights']:
                        if key in attn_raw:
                            w_tensor, _ = _extract_attn(attn_raw[key])
                            spatial_shapes = attn_raw.get('spatial_shapes')
                            return w_tensor, spatial_shapes
                return None, None

            def _attn_to_vis(attn: torch.Tensor, spatial_shapes=None) -> Optional[list[tuple[list[np.ndarray], str]]]:
                """
                将注意力矩阵可视化为若干子图列表。

                支持形状:
                  - [L, H, Q, K]
                  - [H, Q, K]
                  - [Q, K]

                视觉 Query 数常见为 1029 (=28x28+14x14+7x7)；文本 Key 为 2 类。
                这里按每个文本 token 生成一张拼接的分辨率金字塔热力图，便于观察不同尺度的响应。
                """
                t = attn.detach().cpu().float()
                if t.ndim == 4:  # [L, H, Q, K]
                    t = t[-1]   # 取最后一层
                if t.ndim == 3:  # [H, Q, K]
                    t = t.mean(dim=0)  # 按 head 平均 -> [Q, K]
                if t.ndim != 2:
                    return None

                q_len, k_len = t.shape
                # 优先使用 spatial_shapes（若模型返回）拆分 Query
                if spatial_shapes is not None:
                    if torch.is_tensor(spatial_shapes):
                        spatial_shapes = spatial_shapes.cpu().numpy()
                    splits = [(int(h), int(w)) for h, w in spatial_shapes]
                else:
                    splits = [(28, 28), (14, 14), (7, 7)]
                split_sizes = [h * w for h, w in splits]
                if sum(split_sizes) != q_len:
                    # 若长度不匹配，退化为单行可视化
                    vis_list = []
                    for cls in range(min(k_len, 4)):
                        vec = t[:, cls]
                        vmin, vmax = float(vec.min()), float(vec.max())
                        if abs(vmax - vmin) < 1e-8:
                            norm = np.zeros((1, q_len), dtype=np.uint8)
                        else:
                            norm = ((vec - vmin) / (vmax - vmin) * 255.0).clamp(0, 255).byte().unsqueeze(0).numpy()
                        jet = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                        # jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
                        vis_list.append((jet, f'attn_cls{cls}'))
                    return vis_list if vis_list else None

                # 按尺度切片并还原空间排列；仅切分，不在此处拼接，留给外层控制缩放与拼接
                offsets = np.cumsum([0] + split_sizes)
                vis_list: list[tuple[list[np.ndarray], str]] = []
                for cls in range(min(k_len, 4)):  # 最多显示前4个 key
                    tiles = []
                    for (h, w), start, end in zip(splits, offsets[:-1], offsets[1:]):
                        patch = t[start:end, cls].reshape(h, w)
                        vmin, vmax = float(patch.min()), float(patch.max())
                        if abs(vmax - vmin) < 1e-8:
                            norm = np.zeros((h, w), dtype=np.uint8)
                        else:
                            norm = ((patch - vmin) / (vmax - vmin) * 255.0).clamp(0, 255).byte().numpy()
                        jet = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
                        # jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
                        tiles.append(jet)
                    vis_list.append((tiles, f'attn_cls{cls}'))

                return vis_list if vis_list else None

            def _attn_argmax_mask(attn: torch.Tensor, spatial_shapes=None, target_cls: int = 0) -> Optional[list[np.ndarray]]:
                """Generate per-scale masks (no concat) by argmax over attention keys."""
                t = attn.detach().cpu().float()
                if t.ndim == 4:  # [L, H, Q, K]
                    t = t[-1]
                if t.ndim == 3:  # [H, Q, K]
                    t = t.mean(dim=0)
                if t.ndim != 2:
                    return None

                q_len, k_len = t.shape
                if k_len == 0:
                    return None

                if spatial_shapes is not None:
                    if torch.is_tensor(spatial_shapes):
                        spatial_shapes = spatial_shapes.cpu().numpy()
                    splits = [(int(h), int(w)) for h, w in spatial_shapes]
                else:
                    splits = [(28, 28), (14, 14), (7, 7)]

                split_sizes = [h * w for h, w in splits]
                if sum(split_sizes) != q_len:
                    return None

                offsets = np.cumsum([0] + split_sizes)
                tiles: list[np.ndarray] = []
                for (h, w), start, end in zip(splits, offsets[:-1], offsets[1:]):
                    patch = t[start:end, :].reshape(h, w, k_len)
                    argmax = patch.argmax(dim=2)
                    mask = (argmax == target_cls).byte().numpy() * 255
                    mask = np.stack([mask] * 3, axis=-1)
                    tiles.append(mask)

                if len(tiles) == 0:
                    return None

                return tiles

            def _add_title_band(img: np.ndarray, title: str, band_h: int = 28) -> np.ndarray:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                band = np.zeros((band_h, img.shape[1], 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(band, title, (5, band_h - 8), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
                return np.concatenate((band, img), axis=0)

            def _resize_to_target(img: np.ndarray, size=(224, 224)) -> np.ndarray:
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)

            # 统一可视化尺寸：固定 224x224 便于 debug，对齐所有可视化
            vis_target_shape = (224, 224)

            # 基础列：RGB / 预测 / GT
            top_row: list[tuple[np.ndarray, str]] = []
            bottom_row: list[tuple[np.ndarray, str]] = []

            top_row.append((cv2.cvtColor(_resize_to_target(rgb_image_ori, size=vis_target_shape), cv2.COLOR_RGB2BGR), 'RGB'))
            pred_prob_vis = cv2.applyColorMap((pred_binary * 255).astype(np.uint8), cv2.COLORMAP_JET)
            pred_prob_vis = cv2.cvtColor(pred_prob_vis, cv2.COLOR_BGR2RGB)
            top_row.append((_resize_to_target(pred_prob_vis, size=vis_target_shape), 'pred_mask_prob'))

            gt_mask_bw = (gt_binary * 255).astype(np.uint8) if gt_binary is not None else np.zeros_like(pred_binary)
            gt_mask_vis = np.stack([gt_mask_bw] * 3, axis=-1)
            bottom_row.append((_resize_to_target(gt_mask_vis, size=vis_target_shape), 'gt_mask'))

            pred_bw = np.stack([(pred_binary * 255).astype(np.uint8)] * 3, axis=-1)
            bottom_row.append((_resize_to_target(pred_bw, size=vis_target_shape), 'pred_mask'))

            # 追加 score map/OT 结果
            if 'score_map_img' in debug_payload:
                img_vis = _resize_to_target(_to_uint8_img(debug_payload['score_map_img']), size=vis_target_shape)
                top_row.append((img_vis, 'score_map_img'))
                arg_vis = _argmax_to_bw_trav(debug_payload['score_map_img'], traversable_class)
                if arg_vis is not None:
                    bottom_row.append((_resize_to_target(arg_vis, size=vis_target_shape), 'score_map_img_argmax'))
            if 'score_map_sne' in debug_payload:
                sne_vis = _resize_to_target(_to_uint8_img(debug_payload['score_map_sne']), size=vis_target_shape)
                top_row.append((sne_vis, 'score_map_sne'))
                arg_vis = _argmax_to_bw_trav(debug_payload['score_map_sne'], traversable_class)
                if arg_vis is not None:
                    bottom_row.append((_resize_to_target(arg_vis, size=vis_target_shape), 'score_map_sne_argmax'))

            for idx, pi_img in enumerate(debug_payload.get('pi_img_levels', [])):
                pi_vis = _resize_to_target(_to_uint8_img(pi_img), size=vis_target_shape)
                top_row.append((pi_vis, f'pi_img_lvl{idx}'))
                arg_vis = _argmax_to_bw_trav(pi_img, traversable_class)
                if arg_vis is not None:
                    bottom_row.append((_resize_to_target(arg_vis, size=vis_target_shape), f'pi_img_lvl{idx}_argmax'))
            for idx, pi_sne in enumerate(debug_payload.get('pi_sne_levels', [])):
                pi_vis = _resize_to_target(_to_uint8_img(pi_sne), size=vis_target_shape)
                top_row.append((pi_vis, f'pi_sne_lvl{idx}'))
                arg_vis = _argmax_to_bw_trav(pi_sne, traversable_class)
                if arg_vis is not None:
                    bottom_row.append((_resize_to_target(arg_vis, size=vis_target_shape), f'pi_sne_lvl{idx}_argmax'))

            # 注意力可视化：仅保留目标类别的权重图（如可通行类别），放在首行
            if args.save_attn:
                attn_tensor, attn_shapes = _extract_attn(attn_matrix)
                if attn_tensor is not None:
                    target_shape = vis_target_shape
                    attn_vis_list = _attn_to_vis(attn_tensor, spatial_shapes=attn_shapes)
                    if attn_vis_list is not None:
                        target_title = f'attn_cls{traversable_class}'
                        filtered = [(tiles, title) for tiles, title in attn_vis_list if title == target_title]
                        for tiles, title in filtered:
                            # 外层统一缩放并横向拼接
                            resized_tiles = [_resize_to_target(tile, size=target_shape) for tile in tiles]
                            concat_tile = np.concatenate(resized_tiles, axis=1)
                            top_row.insert(0, (concat_tile, title))

                    attn_argmax_tiles = _attn_argmax_mask(attn_tensor, spatial_shapes=attn_shapes, target_cls=traversable_class)
                    if attn_argmax_tiles is not None:
                        resized_tiles = [_resize_to_target(tile, size=target_shape) for tile in attn_argmax_tiles]
                        concat_mask = np.concatenate(resized_tiles, axis=1)
                        bottom_row.insert(0, (concat_mask, 'attn_argmax_mask'))

            if len(top_row) > 0:
                top_titled = [_add_title_band(im, title) for im, title in top_row]
                top_concat = np.concatenate([
                    # cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.shape[2] == 3 else im for im in top_titled
                    im for im in top_titled
                ], axis=1)

                if len(bottom_row) > 0:
                    bottom_titled = [_add_title_band(im, title) for im, title in bottom_row]
                    bottom_concat = np.concatenate([
                        # cv2.cvtColor(im, cv2.COLOR_RGB2BGR) if im.shape[2] == 3 else im for im in bottom_titled
                        im for im in bottom_titled
                    ], axis=1)
                    concat = np.concatenate((top_concat, bottom_concat), axis=0)
                else:
                    concat = top_concat

                cv2.imwrite(str(sample_root / f'{base_name}_debug_concat.png'), concat)

            if args.save_attn and attn_matrix is not None:
                attn_to_save = attn_matrix.detach().cpu() if torch.is_tensor(attn_matrix) else attn_matrix
                torch.save(attn_to_save, str(sample_root / f'{base_name}_attn.pt'))

        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    # mmcv.ProgressBar closes itself when total is reached; no extra finish() call needed

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
