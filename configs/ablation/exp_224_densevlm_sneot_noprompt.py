"""
消融实验 F1a: SNE + OT 融合 (无先验) (ORFD)
- 尺寸: 224x224
- 权重: densevlm_coco
- SNE: 有 (backbone-ot) - 使用最优传输融合
- Prompt: 无
- OT Prior: False (均匀分布)

对比实验: F1a (prior=False) vs F1b (prior=True)
验证使用预测图初始化文本分布的效果
"""
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1k.py',
    '../_base_/datasets/orfd2orfd-224.py']

per_gpu = 16
data = dict(samples_per_gpu=per_gpu, workers_per_gpu=per_gpu)

return_attn = False
class_num = 2
class_thing_num = 0
class_stuff_num = 2
img_size = (224, 224)

# ============================================================================
# 消融配置
# ============================================================================
model_name = 'EVA02-CLIP-B-16'
pretrained_weight = 'weight/pretrained/densevlm_coco_6_save6_512_eva_vib16_12layers.pt'
use_sne = True                # ✅ 启用 SNE
sne_fusion_stage = 'backbone' # backbone 阶段融合 (在 segmentor 中)
sne_fusion_mode = 'ot'        # ✅ 最优传输融合 (对比 proj)
ot_use_score_prior = False    # ❌ 不使用预测图，使用均匀分布
prompt_cls = False
# ============================================================================

_model_dim_map = {
    'EVA01-CLIP-B-16':     {'visual': 768,  'text': 512},
    'EVA02-CLIP-B-16':     {'visual': 768,  'text': 512},
    'EVA02-CLIP-L-14':     {'visual': 1024, 'text': 768},
    'EVA02-CLIP-L-14-336': {'visual': 1024, 'text': 768},
}
visual_feature_dim = _model_dim_map[model_name]['visual']
text_feature_dim = _model_dim_map[model_name]['text']
use_sne_pixel = use_sne and (sne_fusion_stage == 'pixel')
decoder_dim = visual_feature_dim * 2 if (sne_fusion_stage == 'backbone' and sne_fusion_mode == 'concat') else visual_feature_dim

import time
_timestamp = time.strftime('%Y%m%d_%H%M')
exp_name = 'ablation_224_densevlm_sneotFalse_noprompt'

model = dict(
    type='tqt_EVA_CLIP',
    token_embed_dim=text_feature_dim,
    text_dim=text_feature_dim,
    visual_dim=visual_feature_dim,
    context_length=24,
    prefix_text='A region that is ',
    prompt_cls=prompt_cls,
    use_sne=use_sne,
    sne_fusion_stage=sne_fusion_stage,
    sne_fusion_mode=sne_fusion_mode,
    ot_use_score_prior=ot_use_score_prior,
    eva_clip=dict(
        model_name=model_name,
        pretrained=pretrained_weight,
        force_custom_clip=True,
        image_size=img_size,
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
        use_sne_pixel=use_sne_pixel,
        sne_fusion_mode=sne_fusion_mode,
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
            return_attn_weights=return_attn,
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
                        feedforward_channels=visual_feature_dim,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=visual_feature_dim*2,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')),
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
                operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')),
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
            assigner=dict(type='IdentityAssigner', num_cls=class_num),
            sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(
            panoptic_on=True,
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
            filter_low_score=True),
        text_proj=dict(text_in_dim=text_feature_dim, text_out_dim=256)),
    identity_head=dict(
        type='IdentityHead',
        in_channels=1,
        channels=1,
        num_classes=1,
        dropout_ratio=0.1,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
    train_cfg=dict(img_sne_save_path=f'./work_dirs/{exp_name}/{_timestamp}/sne/'),
    test_cfg=dict(
        mode='whole',
        crop_size=(512, 512),
        stride=(341, 341),
        return_attn=return_attn,
        attn_save_dir=f'./work_dirs/{exp_name}/{_timestamp}/attns/'))

optimizer = dict(
    type='AdamW', lr=1e-5, weight_decay=1e-4,
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'text_encoder': dict(lr_mult=0.0),
            'norm': dict(decay_mult=0.)}))

work_dir = f'./work_dirs/{exp_name}/{_timestamp}'
