dist_params = dict(backend='nccl')
log_config = dict(
    interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
find_unused_parameters = True
optimizer_config = dict()
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-06,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06)
runner = dict(type='IterBasedRunner', max_iters=5000)
checkpoint_config = dict(by_epoch=False, interval=1000)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
IMG_MEAN = [122.7709383, 116.7460125, 104.09373615000001]
IMG_VAR = [68.5005327, 66.6321579, 70.32316304999999]
img_norm_cfg = dict(
    mean=[122.7709383, 116.7460125, 104.09373615000001],
    std=[68.5005327, 66.6321579, 70.32316304999999],
    to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(
        type='Normalize',
        mean=[122.7709383, 116.7460125, 104.09373615000001],
        std=[68.5005327, 66.6321579, 70.32316304999999],
        to_rgb=True),
    dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[122.7709383, 116.7460125, 104.09373615000001],
                std=[68.5005327, 66.6321579, 70.32316304999999],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
weather = ['sun']
src_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='training/',
    ann_dir='training/',
    weather=['sun'],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', reduce_zero_label=False),
        dict(type='Resize', ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
        dict(type='RandomFlip', prob=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(
            type='Normalize',
            mean=[122.7709383, 116.7460125, 104.09373615000001],
            std=[68.5005327, 66.6321579, 70.32316304999999],
            to_rgb=True),
        dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
        dict(type='ToMask'),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect',
            keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
    ])
tgt_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='validation/',
    ann_dir='validation/',
    weather=['sun'],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[122.7709383, 116.7460125, 104.09373615000001],
                    std=[68.5005327, 66.6321579, 70.32316304999999],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
test_dataset_dict = dict(
    type='ORFDDataset',
    data_root='dataset/ORFD',
    img_dir='testing/',
    ann_dir='testing/',
    weather=['snow', 'rain', 'fog', 'sun'],
    pipeline=[
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=(2048, 1024),
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(
                    type='Normalize',
                    mean=[122.7709383, 116.7460125, 104.09373615000001],
                    std=[68.5005327, 66.6321579, 70.32316304999999],
                    to_rgb=True),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img'])
            ])
    ])
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type='UGDataset',
        source=dict(
            type='ORFDDataset',
            data_root='dataset/ORFD',
            img_dir='training/',
            ann_dir='training/',
            weather=['sun'],
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', reduce_zero_label=False),
                dict(type='Resize', ratio_range=(0.5, 2.0)),
                dict(
                    type='RandomCrop',
                    crop_size=(512, 512),
                    cat_max_ratio=0.75),
                dict(type='RandomFlip', prob=0.5),
                dict(type='PhotoMetricDistortion'),
                dict(
                    type='Normalize',
                    mean=[122.7709383, 116.7460125, 104.09373615000001],
                    std=[68.5005327, 66.6321579, 70.32316304999999],
                    to_rgb=True),
                dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=255),
                dict(type='ToMask'),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=['img', 'gt_semantic_seg', 'gt_masks', 'gt_labels'])
            ]),
        rare_class_sampling=None),
    val=dict(
        type='ORFDDataset',
        data_root='dataset/ORFD',
        img_dir='validation/',
        ann_dir='validation/',
        weather=['sun'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='ORFDDataset',
        data_root='dataset/ORFD',
        img_dir='testing/',
        ann_dir='testing/',
        weather=['snow', 'rain', 'fog', 'sun'],
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[122.7709383, 116.7460125, 104.09373615000001],
                        std=[68.5005327, 66.6321579, 70.32316304999999],
                        to_rgb=True),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
class_num = 2
class_thing_num = 0
class_stuff_num = 2
text_feature_dim = 512
visual_feature_dim = 768
model = dict(
    type='tqdm_EVA_CLIP',
    token_embed_dim=512,
    text_dim=512,
    context_length=24,
    eva_clip=dict(
        model_name='EVA02-CLIP-B-16',
        pretrained='pretrained/EVA02_CLIP_B_psz16_s8B.pt',
        force_custom_clip=True,
        image_size=512,
        out_indices=[3, 5, 7, 11],
        context_length=32,
        xattn=True),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    decode_head=dict(
        type='tqdmHead',
        in_channels=[768, 768, 768, 768],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_things_classes=0,
        num_stuff_classes=2,
        num_queries=2,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='tqdmMSDeformAttnPixelDecoder',
            num_text_embeds=2,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                type='DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=False,
                            norm_cfg=None,
                            init_cfg=None),
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False)
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=768,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=1536,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
                init_cfg=None),
            positional_encoding=dict(
                type='SinePositionalEncoding', num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(
            type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=9,
            transformerlayers=dict(
                type='DetrTransformerDecoderLayer',
                attn_cfgs=dict(
                    type='MultiheadAttention',
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=False),
                ffn_cfgs=dict(
                    embed_dims=256,
                    feedforward_channels=1536,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=1536,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0, 1.0, 0.1]),
        loss_mask=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict(
            num_points=12544,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(type='IdentityAssigner', num_cls=2),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
            filter_low_score=True),
        text_proj=dict(text_in_dim=512, text_out_dim=256)),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341)))
optimizer = dict(
    type='AdamW',
    lr=1e-05,
    weight_decay=0.0001,
    paramwise_cfg=dict(
        custom_keys=dict(
            backbone=dict(lr_mult=0.1),
            text_encoder=dict(lr_mult=0.0),
            norm=dict(decay_mult=0.0))))
work_dir = './work_dirs_o2o_sun_base/tqdm_eva_vit-b_1e-5_5k-o2o-512-sun-terrian'
gpu_ids = range(0, 1)
