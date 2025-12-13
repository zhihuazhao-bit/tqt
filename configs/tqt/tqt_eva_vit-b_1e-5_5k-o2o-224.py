_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10.py',
    '../_base_/datasets/orfd2orfd-224.py']
per_gpu = 16

data = dict(
    samples_per_gpu=per_gpu,
    workers_per_gpu=per_gpu)

return_attn = False
# return_attn = True  
class_num = 2
class_thing_num = 0
class_stuff_num = 2

# 图像尺寸 (需与 dataset 配置中的 img_size 保持一致)
img_size = (224, 224)

# ============================================================================
# 模型配置 - 特征维度由 model_name 自动决定
# ============================================================================
# | model_name          | visual_feature_dim | text_feature_dim |
# |---------------------|-------------------|------------------|
# | EVA01-CLIP-B-16     | 768               | 512              |
# | EVA02-CLIP-B-16     | 768               | 512              |
# | EVA02-CLIP-L-14     | 1024              | 768              |
# | EVA02-CLIP-L-14-336 | 1024              | 768              |
# ============================================================================
model_name = 'EVA02-CLIP-B-16'

# 根据 model_name 自动确定特征维度
_model_dim_map = {
    'EVA01-CLIP-B-16':     {'visual': 768,  'text': 512},
    'EVA02-CLIP-B-16':     {'visual': 768,  'text': 512},
    'EVA02-CLIP-L-14':     {'visual': 1024, 'text': 768},
    'EVA02-CLIP-L-14-336': {'visual': 1024, 'text': 768},
}
visual_feature_dim = _model_dim_map[model_name]['visual']
text_feature_dim = _model_dim_map[model_name]['text']

# SNE 融合配置
use_sne = False                # 是否使用 SNE 双模态融合
sne_fusion_stage = 'pixel'    # 'backbone' (segmentor中融合) / 'pixel' (head中融合)
sne_fusion_mode = 'proj'      # 'proj' / 'add' / 'concat' / 'cross_attn' / 'ot'

# head 是否需要 pixel 阶段 SNE 融合模块 (仅当 use_sne=True 且 stage='pixel' 时)
use_sne_pixel = use_sne and (sne_fusion_stage == 'pixel')

# Prompt 配置
prompt_cls = False             # 是否使用动态场景感知提示 (天气/光照/路面)

# 当 sne_fusion_stage='backbone' 且 mode='concat' 时，backbone 输出通道翻倍
# 当 sne_fusion_stage='pixel' 且 mode='concat' 时，pixel_decoder 输出通道翻倍
decoder_dim = visual_feature_dim * 2 if (sne_fusion_stage == 'backbone' and sne_fusion_mode == 'concat') else visual_feature_dim

# ============================================================================
# 实验名称和路径 (根据配置自动生成)
# ============================================================================
import time
_timestamp = time.strftime('%Y%m%d_%H%M')
# 模型简称: EVA02-CLIP-B-16 -> eva02_b
_model_short = model_name.lower().replace('eva0', 'eva').replace('-clip-', '_').replace('-16', '').replace('-14', '')
# 实验名称: tqt_{model}_{fusion_stage}_{fusion_mode}[_prompt]_{timestamp}
_prompt_suffix = '_prompt' if prompt_cls else ''
exp_name = f'tqt_{_model_short}_{sne_fusion_stage}_{sne_fusion_mode}{_prompt_suffix}'

model = dict(
    type='tqt_EVA_CLIP',
    token_embed_dim=text_feature_dim,
    text_dim=text_feature_dim,
    visual_dim=visual_feature_dim,
    context_length=24,
    prefix_text = 'A region that is ',
    prompt_cls=prompt_cls,
    use_sne=use_sne,
    sne_fusion_stage=sne_fusion_stage,
    sne_fusion_mode=sne_fusion_mode,    
    eva_clip=dict(
        model_name=model_name,
        # pretrained='weight/pretrained/densevlm_coco_6_save6_512_eva_vib16_12layers.pt',
        pretrained='weight/pretrained/EVA02_CLIP_B_psz16_s8B.pt',
        force_custom_clip=True,
        image_size=img_size,  # 引用 dataset 配置中的 img_size
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
        type='tqtHead',
        in_channels=[decoder_dim, decoder_dim, decoder_dim, decoder_dim],
        feat_channels=256,
        out_channels=256,
        in_index=[0, 1, 2, 3],
        use_sne_pixel=use_sne_pixel,      # 是否启用 pixel 阶段 SNE 融合
        sne_fusion_mode=sne_fusion_mode,  # pixel 阶段融合方式
        num_things_classes=class_thing_num,
        num_stuff_classes=class_stuff_num,
        num_queries=class_num,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type='AttntqdmMSDeformAttnPixelDecoder',
            num_text_embeds=class_num,
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            return_attn_weights=return_attn, # 1. 新增参数
            encoder=dict(
                type='AttnDetrTransformerDecoder' if return_attn else 'DetrTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='AttnDetrTransformerDecoderLayer' if return_attn else 'DetrTransformerDecoderLayer',
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
                            type='AttnMultiheadAttention' if return_attn else 'MultiheadAttention',
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
    train_cfg=dict(
        img_sne_save_path=f'./work_dirs/{exp_name}/{_timestamp}/sne/',
    ),
    test_cfg=dict(
        # mode='slide',
        mode='whole',
        crop_size=(512, 512), 
        stride=(341, 341),
        return_attn=return_attn,
        attn_save_dir=f'./work_dirs/{exp_name}/{_timestamp}/attns/'
        )
)

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-4,
                 paramwise_cfg=dict(
                    custom_keys={
                        'backbone': dict(lr_mult=0.1),
                        'text_encoder': dict(lr_mult=0.0),
                        'norm': dict(decay_mult=0.)}))

work_dir = f'./work_dirs/{exp_name}/{_timestamp}'
