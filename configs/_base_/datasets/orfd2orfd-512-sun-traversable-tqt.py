IMG_MEAN = [ v*255 for v in [0.48145466, 0.4578275, 0.40821073]]
IMG_VAR = [ v*255 for v in [0.26862954, 0.26130258, 0.27577711]]
img_norm_cfg = dict(mean=IMG_MEAN, std=IMG_VAR, to_rgb=True)

crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSneFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # 先对原始图片img和seg随机放大缩小，然后在裁减至crop_size，如果不足512*512，则填充为512*512，
    # 会判断是否和MultiScaleFlipAug结合使用，如果是，则直接执行resize，如果不是则会根据ratio生成scale
    dict(type='Resize', ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    # 代码在mmseg/datasets/pipelines/formating.py, 在这一步对sne、img、seg均进行totensor操作。
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels', 'sne'])]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadSneFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            # 代码在mmseg/datasets/pipelines/formating.py 96, 执行纬度顺序调整，并转化为tensor
            dict(type='ImageToTensor', keys=['img', 'sne']),
            dict(type='Collect', keys=['img', 'sne'])])]

scene_type = 'weather'
scene_scope = ['sunny']
class_names=('notraversable', 'traversable')

src_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='training',
    ann_dir='training',
    scene_type=scene_type,
    scene_scope=scene_scope,
    class_names=class_names,
    pipeline=train_pipeline)
    
tgt_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='validation',
    ann_dir='validation',
    scene_type=scene_type,
    scene_scope=scene_scope,
    class_names=class_names,
    pipeline=test_pipeline)

test_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='testing',
    ann_dir='testing',
    scene_type='weather',
    scene_scope=['sunny', 'snowy', 'foggy', 'rainy'],
    class_names=class_names,
    pipeline=test_pipeline)

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='UGDataset', 
        source=src_dataset_dict,
        rare_class_sampling=None),
    val=tgt_dataset_dict,
    test=test_dataset_dict)