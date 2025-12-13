#!/usr/bin/env python
"""
TQT 消融实验快速验证脚本

验证所有消融配置组合是否能正常运行 (训练 10 iter)

使用方法:
    python verify_ablation.py [--gpu GPU_ID] [--config CONFIG_INDEX]
    
示例:
    python verify_ablation.py              # 运行所有配置
    python verify_ablation.py --gpu 0      # 指定 GPU
    python verify_ablation.py --config 3   # 只运行第 3 个配置
"""

import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from datetime import datetime

# ============================================================================
# 验证配置矩阵
# ============================================================================
ABLATION_CONFIGS = [
    # (use_sne, sne_fusion_stage, sne_fusion_mode, prompt_cls, 描述)
    # backbone 融合 (5 种)
    (True, 'backbone', 'proj',       False, 'backbone + proj'),
    (True, 'backbone', 'add',        False, 'backbone + add'),
    (True, 'backbone', 'concat',     False, 'backbone + concat'),
    (True, 'backbone', 'cross_attn', False, 'backbone + cross_attn'),
    (True, 'backbone', 'ot',         False, 'backbone + ot'),
    # pixel 融合 (5 种)
    (True, 'pixel',    'proj',       False, 'pixel + proj'),
    (True, 'pixel',    'add',        False, 'pixel + add'),
    (True, 'pixel',    'concat',     False, 'pixel + concat'),
    (True, 'pixel',    'cross_attn', False, 'pixel + cross_attn'),
    (True, 'pixel',    'ot',         False, 'pixel + ot'),
    # prompt_cls 测试
    (True, 'pixel',    'proj',       True,  'pixel + proj + prompt_cls'),
    # 无 SNE baseline
    (False, 'pixel',   'proj',       False, 'no_sne (baseline)'),
]

# 基础配置文件路径
BASE_CONFIG = 'configs/tqt/tqt_eva_vit-b_1e-5_5k-r2r-512-all.py'

# 配置文件模板 (需要替换的部分)
CONFIG_TEMPLATE = '''_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10iter.py',
    '../_base_/datasets/road2road-512-all-traversable-cls-tqt-224.py']
per_gpu = 4

data = dict(
    samples_per_gpu=per_gpu,
    workers_per_gpu=4)

return_attn = False
class_num = 2
class_thing_num = 0
class_stuff_num = 2

# ============================================================================
# 模型配置
# ============================================================================
model_name = 'EVA02-CLIP-B-16'

_model_dim_map = {{
    'EVA01-CLIP-B-16':     {{'visual': 768,  'text': 512}},
    'EVA02-CLIP-B-16':     {{'visual': 768,  'text': 512}},
    'EVA02-CLIP-L-14':     {{'visual': 1024, 'text': 768}},
    'EVA02-CLIP-L-14-336': {{'visual': 1024, 'text': 768}},
}}
visual_feature_dim = _model_dim_map[model_name]['visual']
text_feature_dim = _model_dim_map[model_name]['text']

# ============================================================================
# 【验证配置】
# ============================================================================
use_sne = {use_sne}
sne_fusion_stage = '{sne_fusion_stage}'
sne_fusion_mode = '{sne_fusion_mode}'
prompt_cls = {prompt_cls}

decoder_dim = visual_feature_dim * 2 if (sne_fusion_stage == 'backbone' and sne_fusion_mode == 'concat') else visual_feature_dim

exp_name = 'verify_ablation'

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
    eva_clip=dict(
        model_name=model_name,
        pretrained='weight/pretrained/densevlm_coco_6_save6_512_eva_vib16_12layers.pt',
        force_custom_clip=True,
        image_size=(224,224),
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
                            init_cfg=None
                        ),
                        dict(
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
                        feedforward_channels=visual_feature_dim,
                        num_fcs=2,
                        act_cfg=dict(type='ReLU', inplace=True),
                        ffn_drop=0.0,
                        dropout_layer=None,
                        add_identity=True),
                    feedforward_channels=visual_feature_dim*2,
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
            semantic_on=False,
            instance_on=True,
            max_per_image=100,
            iou_thr=0.8,
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
        img_sne_save_path='./work_dirs/verify_ablation/sne/',
    ),
    test_cfg=dict(
        mode='whole',
        crop_size=(512, 512), 
        stride=(341, 341),
        return_attn=return_attn,
        attn_save_dir='./work_dirs/verify_ablation/attns/'
        )
)

optimizer = dict(type='AdamW', lr=1e-5, weight_decay=1e-4,
                 paramwise_cfg=dict(
                    custom_keys={{
                        'backbone': dict(lr_mult=0.1),
                        'text_encoder': dict(lr_mult=0.0),
                        'norm': dict(decay_mult=0.)}}))

work_dir = './work_dirs/verify_ablation'
'''


def generate_config(use_sne, sne_fusion_stage, sne_fusion_mode, prompt_cls):
    """生成临时配置文件内容"""
    return CONFIG_TEMPLATE.format(
        use_sne=use_sne,
        sne_fusion_stage=sne_fusion_stage,
        sne_fusion_mode=sne_fusion_mode,
        prompt_cls=prompt_cls
    )


def run_training(config_path, gpu_id=0):
    """运行训练命令"""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python train.py --config {config_path}"
    result = subprocess.run(
        cmd, 
        shell=True, 
        capture_output=True, 
        text=True,
        cwd='/root/tqdm'
    )
    return result.returncode == 0, result.stdout, result.stderr


def print_banner(text, char='='):
    """打印横幅"""
    width = 70
    print(f"\n{char * width}")
    print(f" {text}")
    print(f"{char * width}")


def main():
    parser = argparse.ArgumentParser(description='TQT 消融实验快速验证')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=int, default=None, 
                        help='只运行指定索引的配置 (0-based)')
    parser.add_argument('--list', action='store_true', help='列出所有配置')
    args = parser.parse_args()
    
    # 列出配置
    if args.list:
        print("\n可用的验证配置:")
        print("-" * 70)
        for i, (use_sne, stage, mode, prompt, desc) in enumerate(ABLATION_CONFIGS):
            print(f"  [{i:2d}] {desc:30s} | sne={use_sne}, stage={stage}, mode={mode}, prompt={prompt}")
        print("-" * 70)
        return
    
    # 确定要运行的配置
    if args.config is not None:
        configs_to_run = [args.config]
    else:
        configs_to_run = list(range(len(ABLATION_CONFIGS)))
    
    print_banner("TQT 消融实验快速验证")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {args.gpu}")
    print(f"配置数量: {len(configs_to_run)}")
    
    # 在 configs/tqt 目录下创建临时配置文件 (保持 _base_ 相对路径正确)
    temp_dir = os.path.join('/root/tqdm', 'configs', 'tqt')
    print(f"配置目录: {temp_dir}")
    
    results = []
    
    try:
        for idx in configs_to_run:
            use_sne, stage, mode, prompt, desc = ABLATION_CONFIGS[idx]
            
            print_banner(f"[{idx+1}/{len(ABLATION_CONFIGS)}] {desc}", '-')
            print(f"  use_sne:          {use_sne}")
            print(f"  sne_fusion_stage: {stage}")
            print(f"  sne_fusion_mode:  {mode}")
            print(f"  prompt_cls:       {prompt}")
            
            # 生成配置文件
            config_content = generate_config(use_sne, stage, mode, prompt)
            config_path = os.path.join(temp_dir, f'verify_config_{idx}.py')
            with open(config_path, 'w') as f:
                f.write(config_content)
            
            # 运行训练
            print(f"\n  正在验证...")
            success, stdout, stderr = run_training(config_path, args.gpu)
            
            if success:
                print(f"  ✓ 验证通过")
                results.append((idx, desc, 'PASS', ''))
            else:
                # 提取错误信息
                error_lines = stderr.strip().split('\n')[-10:]  # 最后 10 行
                error_msg = '\n'.join(error_lines)
                print(f"  ✗ 验证失败")
                print(f"  错误信息:")
                for line in error_lines:
                    print(f"    {line}")
                results.append((idx, desc, 'FAIL', error_msg))
    
    finally:
        # 保留配置目录，不清理 (方便调试)
        print(f"\n配置文件保留在: {temp_dir}")
    
    # 打印总结
    print_banner("验证结果总结")
    
    passed = sum(1 for r in results if r[2] == 'PASS')
    failed = sum(1 for r in results if r[2] == 'FAIL')
    
    print(f"\n{'序号':<5} {'配置':<35} {'结果':<10}")
    print("-" * 55)
    for idx, desc, status, _ in results:
        symbol = '✓' if status == 'PASS' else '✗'
        print(f"{idx:<5} {desc:<35} {symbol} {status}")
    
    print("-" * 55)
    print(f"通过: {passed}/{len(results)}, 失败: {failed}/{len(results)}")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 返回状态码
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
