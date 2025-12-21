IMG_MEAN = [v * 255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [v * 255 for v in [0.26862954, 0.26130258, 0.27577711]]
img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

# Image size for 1024x1024 experiments
img_size = (1024, 1024)

# Reduce batch size for the larger resolution to avoid OOM
per_gpu = 4

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSneFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', img_scale=img_size, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels', 'sne'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSneFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_size,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img', 'sne']),
            dict(type='Collect', keys=['img', 'sne'])])]

scene_type = 'road'
scene_scope = ['paved', 'unpaved']
class_names = ('notraversable', 'traversable')

src_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/road3d',
    img_dir='training',
    ann_dir='training',
    scene_type=scene_type,
    scene_scope=scene_scope,
    class_names=class_names,
    pipeline=train_pipeline)
    
tgt_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/road3d',
    img_dir='validation',
    ann_dir='validation',
    scene_type=scene_type,
    scene_scope=scene_scope,
    class_names=class_names,
    pipeline=test_pipeline)

int_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/road3d',
    img_dir='testing',
    ann_dir='testing',
    scene_type='road',
    scene_scope=['paved', 'unpaved'],
    class_names=class_names,
    pipeline=test_pipeline)

test_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/road3d',
    img_dir='testing',
    ann_dir='testing',
    scene_type='road',
    scene_scope=['paved', 'unpaved'],
    class_names=class_names,
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=per_gpu,
    workers_per_gpu=per_gpu,
    train=dict(
        type='UGDataset',
        source=src_dataset_dict,
        rare_class_sampling=None),
    val=tgt_dataset_dict,
    test=test_dataset_dict,
    int=int_dataset_dict)
