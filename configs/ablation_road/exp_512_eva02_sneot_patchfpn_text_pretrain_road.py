"""
消融实验 #6: SNE OT + Patch-FPN + PromptSoft + Text Pretrain Only (Road3D)
- 基于: exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_config_prob_softunion_road_learnableT_promptTau.py
- text_pretrain=True, image_pretrain=False
- prompt_cls=True (保持原配置的场景解耦)
"""
_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_5k.py',
    '../_base_/datasets/road2road-512.py']

per_gpu = 16
data = dict(samples_per_gpu=per_gpu, workers_per_gpu=per_gpu)

return_attn = False
class_num = 2
class_thing_num = 0
class_stuff_num = 2
img_size = (512, 512)
tau = None

# ============================================================================
model_name = 'EVA02-CLIP-B-16'
pretrained_weight = 'weight/pretrained/EVA02_CLIP_B_psz16_s8B.pt'
load_text_pretrained = True
load_image_pretrained = False
use_sne = True
sne_fusion_stage = 'backbone'
sne_fusion_mode = 'ot'
ot_use_score_prior = True
ot_score_prior_mode = 'prob'
ot_score_prior_temperature = None
ot_cost_type = 'cos'
ot_fuse_output = False
ot_fuse_mode = 'config'
ot_softunion = True
prompt_cls = True
prompt_cls_mode = 'soft'
prompt_cls_temperature_mode = 'tau'
prompt_cls_topk_train = None
prompt_cls_topk_test = None
patch_fpn = True
supervise_ot_pi = True
# ============================================================================

_model_dim_map = {'EVA02-CLIP-B-16': {'visual': 768, 'text': 512}}
visual_feature_dim = _model_dim_map[model_name]['visual']
text_feature_dim = _model_dim_map[model_name]['text']
decoder_dim = visual_feature_dim

import time
_timestamp = time.strftime('%Y%m%d_%H%M')
exp_name = 'ablation_512_eva02_sneot_patchfpn_text_pretrain_road'

model = dict(
    type='tqt_EVA_CLIP',
    token_embed_dim=text_feature_dim, text_dim=text_feature_dim, visual_dim=visual_feature_dim,
    context_length=24, prefix_text='A region that is ',
    prompt_cls=prompt_cls, prompt_cls_mode=prompt_cls_mode, prompt_cls_temperature_mode=prompt_cls_temperature_mode, prompt_cls_topk_train=prompt_cls_topk_train, prompt_cls_topk_test=prompt_cls_topk_test,
    use_sne=use_sne, sne_fusion_stage=sne_fusion_stage, sne_fusion_mode=sne_fusion_mode,
    ot_use_score_prior=ot_use_score_prior, ot_score_prior_mode=ot_score_prior_mode, ot_score_prior_temperature=ot_score_prior_temperature, ot_cost_type=ot_cost_type, ot_fuse_output=ot_fuse_output, ot_fuse_mode=ot_fuse_mode, ot_softunion=ot_softunion,
    patch_fpn=patch_fpn, supervise_ot_pi=supervise_ot_pi, tau=tau,
    eva_clip=dict(model_name=model_name, pretrained=pretrained_weight, load_text_pretrained=load_text_pretrained, load_image_pretrained=load_image_pretrained, force_custom_clip=True, image_size=img_size, out_indices=[3, 5, 7, 11], context_length=32, xattn=True),
    context_decoder=dict(type='ContextDecoder', transformer_width=256, transformer_heads=4, transformer_layers=3, visual_dim=text_feature_dim, dropout=0.1, outdim=text_feature_dim, style='pytorch'),
    decode_head=dict(
        type='tqtHead', in_channels=[decoder_dim]*4, feat_channels=256, out_channels=256, in_index=[0, 1, 2, 3],
        use_sne_pixel=False, sne_fusion_mode=sne_fusion_mode, num_things_classes=class_thing_num, num_stuff_classes=class_stuff_num, num_queries=class_num, num_transformer_feat_level=3,
        pixel_decoder=dict(type='AttntqdmMSDeformAttnPixelDecoder', num_text_embeds=class_num, num_outs=3, norm_cfg=dict(type='GN', num_groups=32), act_cfg=dict(type='ReLU'), return_attn_weights=return_attn,
            encoder=dict(type='DetrTransformerDecoder', return_intermediate=True, num_layers=6, transformerlayers=dict(type='DetrTransformerDecoderLayer', attn_cfgs=[dict(type='MultiScaleDeformableAttention', embed_dims=256, num_heads=8, num_levels=3, num_points=4, im2col_step=64, dropout=0.0, batch_first=False), dict(type='MultiheadAttention', embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, batch_first=False)], ffn_cfgs=dict(embed_dims=256, feedforward_channels=visual_feature_dim, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0), feedforward_channels=visual_feature_dim*2, operation_order=('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')), init_cfg=None),
            positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True), init_cfg=None),
        enforce_decoder_input_project=False, positional_encoding=dict(type='SinePositionalEncoding', num_feats=128, normalize=True),
        transformer_decoder=dict(type='DetrTransformerDecoder', return_intermediate=True, num_layers=9, transformerlayers=dict(type='DetrTransformerDecoderLayer', attn_cfgs=dict(type='MultiheadAttention', embed_dims=256, num_heads=8, attn_drop=0.0, proj_drop=0.0, batch_first=False), ffn_cfgs=dict(embed_dims=256, feedforward_channels=visual_feature_dim*2, num_fcs=2, act_cfg=dict(type='ReLU', inplace=True), ffn_drop=0.0), feedforward_channels=visual_feature_dim*2, operation_order=('cross_attn', 'norm', 'self_attn', 'norm', 'ffn', 'norm')), init_cfg=None),
        loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0, reduction='mean', class_weight=[1.0]*class_num+[0.1]),
        loss_mask=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=5.0),
        loss_dice=dict(type='DiceLoss', use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0, loss_weight=5.0),
        train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, assigner=dict(type='IdentityAssigner', num_cls=class_num), sampler=dict(type='MaskPseudoSampler')),
        test_cfg=dict(panoptic_on=True, semantic_on=False, instance_on=True, max_per_image=100, iou_thr=0.8, filter_low_score=True),
        text_proj=dict(text_in_dim=text_feature_dim, text_out_dim=256)),
    identity_head=dict(type='IdentityHead', in_channels=1, channels=1, num_classes=1, dropout_ratio=0.1, align_corners=False, loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.)),
    train_cfg=dict(img_sne_save_path=None, debug_vis=dict(max_samples=5, interval=1000, save_debug=True, output_dir=None)),
    test_cfg=dict(mode='whole', crop_size=(512, 512), stride=(341, 341), return_attn=return_attn, attn_save_dir=f'./work_dirs/{exp_name}/{_timestamp}/attns/'))

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-4, paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1), 'text_encoder': dict(lr_mult=0.0), 'norm': dict(decay_mult=0.)}))
work_dir = f'./work_dirs/{exp_name}/{_timestamp}'
custom_hooks = [dict(type='SetIterHook', priority='VERY_HIGH'), dict(type='SaveTrainVisHook', priority='HIGH', interval=100, max_samples=50, save_debug=True, output_dir=None)]
