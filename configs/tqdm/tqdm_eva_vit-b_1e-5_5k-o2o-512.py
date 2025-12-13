_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10.py',
    '../_base_/datasets/orfd2orfd-512.py']

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8)

# 实验名称和路径 (根据配置自动生成)
import time
_timestamp = time.strftime('%Y%m%d_%H%M')
exp_name = 'tqdm_eva2_b_o2o_512'

class_num = 2
class_thing_num = 0
class_stuff_num = 2
text_feature_dim = 512
visual_feature_dim = 768
image_size = (512, 512)
model = dict(
    type='tqdm_EVA_CLIP',
    token_embed_dim=text_feature_dim,
    text_dim=text_feature_dim,
    context_length=24,
    eva_clip=dict(
        model_name='EVA02-CLIP-B-16',
        pretrained='weight/pretrained/EVA02_CLIP_B_psz16_s8B.pt',
        force_custom_clip=True,
        image_size=image_size,
        out_indices=[3, 5, 7, 11],
        context_length=32,
        xattn=True),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=text_feature_dim,
        dropout=0.1,
        outdim=text_feature_dim,
        style='pytorch'),
    decode_head=dict(
        type='tqdmHead',
        in_channels=[visual_feature_dim, visual_feature_dim, visual_feature_dim, visual_feature_dim],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        num_things_classes=class_thing_num,
        num_stuff_classes=class_stuff_num,
        num_queries=class_num,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='tqdmMSDeformAttnPixelDecoder',
            num_text_embeds=class_num,
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
                        dict( # for self attention
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_heads=8,
                            num_levels=3,
                            num_points=4,
                            im2col_step=64,
                            dropout=0.0,
                            batch_first=False,
                            norm_cfg=None, 
                            init_cfg=None
                        ),
                        dict( # for cross attention
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            attn_drop=0.0,
                            proj_drop=0.0,
                            dropout_layer=None,
                            batch_first=False
                        )
                    ],
                    ffn_cfgs=dict(
                        embed_dims=256,
                        feedforward_channels=visual_feature_dim, # reduced from 2048,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=visual_feature_dim*2,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                    'ffn', 'norm')), # same order with mask2former / 
                    # operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                    #                'ffn', 'norm')), # LGFormer order.
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
                    feedforward_channels=visual_feature_dim*2,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True),
                feedforward_channels=visual_feature_dim*2,
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm',
                                 'ffn', 'norm')),
            init_cfg=None),
        loss_cls=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * class_num + [0.1]),
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
            assigner=dict(
                type='IdentityAssigner',
                num_cls=class_num,),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            # For now, the dataset does not support
            # evaluating semantic segmentation metric.
            semantic_on=False,
            instance_on=True,
            # max_per_image is for instance segmentation.
            max_per_image=100,
            iou_thr=0.8,
            # In Mask2Former's panoptic postprocessing,
            # it will filter mask area where score is less than 0.5 .
            filter_low_score=True), 
        text_proj=dict(
            text_in_dim=text_feature_dim,
            text_out_dim=256),
    ),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole', crop_size=(512, 512), stride=(341, 341))
)

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-4,
                 paramwise_cfg=dict(
                    custom_keys={
                        'backbone': dict(lr_mult=0.1),
                        'text_encoder': dict(lr_mult=0.0),
                        'norm': dict(decay_mult=0.)}))

work_dir = f'./work_dirs/{exp_name}/{_timestamp}'
