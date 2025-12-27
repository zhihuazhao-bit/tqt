# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TQDM (Textual Query-Driven Mask Transformer) is an ECCV 2024 research project for domain-generalized semantic segmentation using vision-language models. Built on MMSegmentation, it uses CLIP and EVA02-CLIP as visual and text encoders.

## Common Commands

### Training
```bash
# Single GPU
python train.py --config configs/tqdm/tqdm_eva_vit-l_1e-5_20k-g2c-512.py --gpus 1 --seed 2023

# Distributed (4 GPUs)
bash dist_train.sh configs/tqdm/tqdm_eva_vit-l_1e-5_20k-g2c-512.py 4

# Resume from checkpoint
python train.py --config CONFIG --resume-from work_dirs/.../latest.pth
```

### Testing
```bash
# Single GPU
python test.py --config CONFIG --checkpoint CKPT --eval mIoU mFscore --show-dir ./results/

# Distributed
bash dist_test.sh CONFIG CKPT 4 --eval mIoU

# Multi-scale flip augmentation
python test.py --config CONFIG --checkpoint CKPT --aug-test --eval mIoU
```

### Ablation Experiments
```bash
# Run ablation tests
bash test_all_ablation.sh      # ORFD dataset
bash test_all_ablation_road.sh # Road3D dataset

# Aggregate results
python statis_ablation_results.py
```

### Dataset Conversion
```bash
python tools/convert_datasets/gta.py data/gta --nproc 8
python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8
```

## Architecture

### Model Hierarchy
```
models/segmentors/
├── tqdm_clip.py      # CLIP ViT-B based segmentor
├── tqdm_eva_clip.py  # EVA02-CLIP segmentor
└── tqt_eva_clip.py   # Latest TQT model with ablation support
```

### Key Components
- **Backbone**: EVA-CLIP or CLIP-ViT vision encoder (`models/backbones/eva_clip/`, `models/backbones/clip/`)
- **Text Encoder**: Frozen CLIP text encoder
- **Decode Head**: DETR-style mask transformer (`mmseg/models/decode_heads/tqdm_head.py`)
- **Pixel Decoder**: Text-to-pixel attention (`mmseg/models/plugins/tqdm_msdeformattn_pixel_decoder.py`)
- **Assigner**: Fixed matching for query-label assignment (`mmseg/models/utils/assigner.py`)

### TQT Model (tqt_eva_clip.py) Ablation Parameters
- `use_sne`: Surface Normal Estimation fusion
- `sne_fusion_mode`: proj, add, concat, cross_attn, ot
- `ot_fuse_mode`: proj, mean, max, config (entropy-based)
- `prompt_cls`: Dynamic scene-aware prompting
- `prompt_cls_type`: text_encoder, linear, linear_text
- `prompt_cls_topk_train/test`: TopK filtering for joint probability
- `use_context_decoder`: Context decoder for text embedding
- `use_learnable_prompt`: Learnable CoOp-style prompts

## Configuration System

### Config Structure
```
configs/
├── _base_/
│   ├── default_runtime.py
│   ├── schedules/          # Training schedules (5k-80k iterations)
│   └── datasets/           # Dataset pipelines (gta2city, road2road, orfd2orfd)
├── tqdm/                   # Main model configs
├── ablation/               # ORFD ablation configs
└── ablation_road/          # Road3D ablation configs
```

### Config Pattern
```python
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py',
    '../_base_/datasets/gta2city-512.py']

model = dict(
    type='tqt_EVA_CLIP',  # or tqdm_CLIP, tqdm_EVA_CLIP
    eva_clip=dict(model_name='EVA02-CLIP-B-16', ...),
    decode_head=dict(type='tqtHead', ...),
    ...
)
```

## Work Directory Structure
```
work_dirs/
└── {exp_name}/
    └── {timestamp}/
        └── {exp_name}/
            ├── best_mIoU_iter_XXXX.pth
            ├── iter_XXXX.pth
            └── {timestamp}.log
```

## Key Dependencies
- PyTorch 2.0+ with CUDA
- MMCV-full 1.5.3 or 1.7.2
- xformers 0.0.20+
- swanlab (experiment tracking)

## Pre-trained Models
Place in `./pretrained/`:
- `CLIP-ViT-B-16.pt` (OpenAI CLIP)
- `EVA02_CLIP_L_336_psz14_s6B.pt` (EVA02-CLIP)
- `EVA02_CLIP_B_psz16_s8B.pt` (EVA02-CLIP-B)

## Supported Datasets
- GTA5, Cityscapes, BDD100K, Mapillary, SYNTHIA (urban scenes)
- ORFD (off-road following)
- Road3D (road scenes)

Dataset paths are configured in `configs/_base_/datasets/`.
