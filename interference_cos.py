import argparse
import os
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.parallel import scatter
from mmcv.runner import load_checkpoint
from matplotlib import pyplot as plt

from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset

import models  # 导入自定义模型注册


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Extract cosine-similarity maps between image patches and text prompts.')
    parser.add_argument('--config', default="/root/tqdm/configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj.py", help='Path to the mmseg config file.')
    parser.add_argument('--checkpoint', default="/root/tqdm/work_dirs/tqt_eva_vit-b_1e-5_5k-o2o-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj/best_mIoU_iter_1000.pth", help='Model checkpoint for inference.')
    parser.add_argument('--output-dir', default='./work_dirs/similarity_maps', help='Directory to save similarity maps.')
    parser.add_argument('--device', default='cuda:0', help='Device for inference, e.g. "cuda:0" or "cpu".')
    parser.add_argument('--limit', type=int, default=-1, help='Maximum number of frames to process (-1 for all).')
    parser.add_argument('--progress', action='store_true', help='Show a progress bar during processing.')
    parser.add_argument('--show-dir', default=None, help='Directory to save visualizations (overrides output-dir).')
    return parser.parse_args()


def _slugify(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch.lower())
        else:
            keep.append('_')
    slug = ''.join(keep).strip('_')
    return slug or 'class'



def main() -> None:
    args = parse_args()
    cfg = Config.fromfile(args.config)

    from models.backbones.utils import SimpleTokenizer, tokenize

    tokenizer = SimpleTokenizer()
    tokens = np.zeros((len(cfg.class_names)), dtype=np.int64)
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
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

    model = model.to(device)
    model.eval()

    class_prompts = [
        "a region that is traversable",
        "a region that is untraversable"
    ]
    class_names = [
        "traversable",
        "untraversable"
    ]
    context_length = model.text_encoder.context_length
    traversable_token = tokenize(class_prompts[0], context_length=context_length).to(device)
    notraversable_token = tokenize(class_prompts[1], context_length=context_length).to(device)

    contexts_traversable = getattr(model, 'contexts_traversable', None)
    contexts_notraversable = getattr(model, 'contexts_notraversable', None)

    with torch.no_grad():
        traversable_embedding = model.text_encoder(
            traversable_token,
            context=contexts_traversable
        )
        notraversable_embedding = model.text_encoder(
            notraversable_token,
            context=contexts_notraversable
        )
        text_features = torch.cat(
            [traversable_embedding, notraversable_embedding],
            dim=0
        )
        text_features = F.normalize(text_features, dim=-1, p=2)

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

    processed = 0

    for data in data_loader:
        if 0 <= args.limit <= processed:
            break

        scattered = scatter(data, [device])[0]
        imgs = scattered['img']
        if isinstance(imgs, (list, tuple)):
            imgs = imgs[0]
        img_metas = scattered['img_metas']
        if isinstance(img_metas, (list, tuple)):
            img_meta_container = img_metas[0]
        else:
            img_meta_container = img_metas
        if isinstance(img_meta_container, (list, tuple)):
            img_meta = img_meta_container[0]
        else:
            img_meta = img_meta_container

        filename = img_meta.get('ori_filename', img_meta.get('filename', f'{processed:06d}.png'))
        if os.path.isabs(filename):
            rgb_path = filename
        else:
            img_prefix = img_meta.get('img_prefix', getattr(dataset, 'img_dir', None))
            if img_prefix:
                rgb_path = os.path.join(img_prefix, filename)
            else:
                rgb_path = filename

        path_parts = Path(rgb_path).parts
        base_name = Path(rgb_path).stem
        scene_name = 'unknown'
        for i, part in enumerate(path_parts):
            if part in ['testing', 'training', 'validation'] and i + 1 < len(path_parts):
                scene_name = path_parts[i + 1]
                break

        rgb_image = mmcv.imread(str(rgb_path), channel_order='rgb')

        with torch.no_grad():
            feats = model.backbone.extract_feats(imgs)
            _, visual_embeddings = feats[-1]
            visual_embeddings = F.normalize(visual_embeddings, dim=1, p=2)
            similarity_map = torch.einsum('bchw,kc->bkhw', visual_embeddings, text_features)
            ori_h, ori_w = img_meta['ori_shape'][:2]
            similarity_map = F.interpolate(
                similarity_map,
                size=(ori_h, ori_w),
                mode='bilinear',
                align_corners=False
            )

        similarity_np = similarity_map.squeeze(0).cpu().numpy()

        save_dir = output_root / scene_name
        save_dir.mkdir(parents=True, exist_ok=True)

        for class_idx, class_name in enumerate(class_names):
            class_map = similarity_np[class_idx]
            heat = (class_map - class_map.min()) / (class_map.max() - class_map.min() + 1e-6)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(rgb_image)
            im = ax.imshow(
                heat,
                cmap='inferno',
                alpha=0.5,
                interpolation='bilinear'
            )
            ax.set_title(class_name)
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            safe_class = _slugify(class_name)
            svg_path = save_dir / f'sim_{base_name}_{safe_class}.svg'
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            plt.close(fig)

        processed += 1
        if progress_bar is not None:
            progress_bar.update(1)

    if progress_bar is not None:
        progress_bar.bar.finish()

    print(f'Processed {processed} images')
    print(f'Similarity maps saved to: {output_root}')


if __name__ == '__main__':
    main()
