import argparse
from pathlib import Path
from typing import Dict, Optional

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint

from mmseg.apis import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run inference on a single scene and create 2x2 previews.')
    parser.add_argument('--config', required=True, help='Path to the mmseg config file.')
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint for inference.')
    parser.add_argument('--scene-dir', required=True, help='Scene directory containing image/normal/label folders.')
    parser.add_argument('--output-dir', default='./work_dirs/scene_preview', help='Directory to save visualization results.')
    parser.add_argument('--image-dirname', default='image data', help='Sub-folder containing RGB images.')
    parser.add_argument('--normal-dirname', default='normal data', help='Sub-folder containing normal maps.')
    parser.add_argument('--label-dirname', default='label', help='Sub-folder containing label maps.')
    parser.add_argument('--device', default='cuda:0', help='Device for inference, e.g. "cuda:0" or "cpu".')
    parser.add_argument('--alpha', type=float, default=0.6, help='Overlay weight for prediction vs. original image.')
    parser.add_argument('--limit', type=int, default=-1, help='Maximum number of frames to process (-1 for all).')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar during processing.')
    return parser.parse_args()


def _patch_dataset_cfg(cfg: Dict, args: argparse.Namespace) -> Dict:
    def _apply(sub_cfg: Dict) -> Dict:
        updated = sub_cfg.copy()
        if 'data_root' in updated:
            updated['data_root'] = args.scene_dir
        if 'img_dir' in updated:
            updated['img_dir'] = args.image_dirname
        if 'img_prefix' in updated:
            updated['img_prefix'] = str(Path(args.scene_dir) / args.image_dirname)
        if 'ann_dir' in updated:
            updated['ann_dir'] = args.label_dirname
        if 'sne_dir' in updated:
            updated['sne_dir'] = args.normal_dirname
        if 'split' in updated:
            updated['split'] = None
        for key in ('dataset', 'datasets'):
            if key in updated:
                if isinstance(updated[key], list):
                    updated[key] = [_apply(item) for item in updated[key]]
                elif isinstance(updated[key], dict):
                    updated[key] = _apply(updated[key])
        return updated

    return _apply(cfg)


def _ensure_palette(model, dataset) -> np.ndarray:
    palette = getattr(model, 'PALETTE', None)
    if palette is None:
        palette = getattr(dataset, 'PALETTE', None)
    if palette is None:
        palette = [[0, 0, 0], [255, 255, 255]]
    return np.array(palette, dtype=np.uint8)


def _indices_to_color(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    max_label = int(mask.max()) if mask.size else 0
    if max_label >= palette.shape[0]:
        extra = palette[-1][None, :]
        repeat = max_label - palette.shape[0] + 1
        palette = np.concatenate([palette, np.repeat(extra, repeat, axis=0)], axis=0)
    return palette[mask]


def _to_display(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[:2] != target_shape:
        img = cv2.resize(img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR)
    return img


def _normal_to_rgb(normal: Optional[np.ndarray], target_shape: tuple[int, int]) -> np.ndarray:
    if normal is None:
        return np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)
    normal = normal.astype(np.float32)
    vmin, vmax = normal.min(), normal.max()
    if vmax - vmin < 1e-6:
        normal[:] = 0
    else:
        normal = (normal - vmin) / (vmax - vmin)
    normal = (normal * 255).clip(0, 255).astype(np.uint8)
    return _to_display(normal, target_shape)


def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test = _patch_dataset_cfg(cfg.data.test, args)
    cfg.data.test.setdefault('test_mode', True)

    dataset = build_dataset(cfg.data.test)
    workers = cfg.data.get('workers_per_gpu', 0)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers,
        dist=False,
        shuffle=False
    )

    device = torch.device(args.device)
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES

    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        model.PALETTE = getattr(dataset, 'PALETTE', None)

    model.to(device)
    model.eval()

    palette = _ensure_palette(model, dataset)
    scene_name = Path(args.scene_dir).name
    output_root = Path(args.output_dir) / scene_name
    output_root.mkdir(parents=True, exist_ok=True)

    total = len(dataset)
    if args.limit >= 0:
        total = min(total, args.limit)
    progress_bar = mmcv.ProgressBar(total) if args.progress else None

    processed = 0
    for data in data_loader:
        if 0 <= args.limit <= processed:
            break

        img_meta = data['img_metas'].data[0][0]
        img_tensor = data['img'].data[0].to(device)

        sne_tensor = data.get('sne')
        if sne_tensor is not None:
            sne_tensor = sne_tensor.data[0].to(device)

        with torch.no_grad():
            if sne_tensor is not None:
                result = model.simple_test(img_tensor, [img_meta], rescale=True, sne=sne_tensor)
            else:
                result = model.simple_test(img_tensor, [img_meta], rescale=True)

        seg_pred = result[0].astype(np.uint8)
        base_path = Path(img_meta.get('ori_filename', img_meta['filename']))
        base_name = base_path.stem

        rgb_img = mmcv.imread(str(base_path), channel_order='rgb')
        target_shape = rgb_img.shape[:2]
        rgb_display = _to_display(rgb_img, target_shape)

        normal_path = img_meta.get('sne_filename')
        if normal_path is None:
            normal_path = Path(args.scene_dir) / args.normal_dirname / (base_name + base_path.suffix)
        normal_path = Path(normal_path)
        normal_img = mmcv.imread(str(normal_path), flag='unchanged') if normal_path.exists() else None
        normal_display = _normal_to_rgb(normal_img, target_shape)

        gt_tensor = data.get('gt_semantic_seg')
        if gt_tensor is not None:
            label_mask = gt_tensor.data[0].squeeze().cpu().numpy().astype(np.uint8)
            gt_color = _indices_to_color(label_mask, palette)
        else:
            label_path = Path(args.scene_dir) / args.label_dirname / (base_name + base_path.suffix)
            if label_path.exists():
                label_img = mmcv.imread(str(label_path), flag='unchanged')
                if label_img.ndim == 2:
                    gt_color = _indices_to_color(label_img.astype(np.uint8), palette)
                else:
                    gt_color = _to_display(label_img[..., :3], target_shape)
            else:
                gt_color = np.zeros((target_shape[0], target_shape[1], 3), dtype=np.uint8)

        pred_color = _indices_to_color(seg_pred, palette)
        pred_color = _to_display(pred_color, target_shape)
        gt_color = _to_display(gt_color, target_shape)
        normal_display = _to_display(normal_display, target_shape)
        overlay = (args.alpha * pred_color + (1.0 - args.alpha) * rgb_display).astype(np.uint8)

        top_row = np.concatenate([rgb_display, normal_display], axis=1)
        bottom_row = np.concatenate([overlay, gt_color], axis=1)
        collage = np.concatenate([top_row, bottom_row], axis=0)
        collage_bgr = cv2.cvtColor(collage, cv2.COLOR_RGB2BGR)

        save_path = output_root / f'{base_name}.png'
        cv2.imwrite(str(save_path), collage_bgr)

        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.bar.finish()

    print(f'Done. Saved {processed} preview images to {output_root}.')


if __name__ == '__main__':
    main()
