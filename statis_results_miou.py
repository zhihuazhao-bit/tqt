#!/usr/bin/env python3
"""
消融实验结果统计脚本 (仅 mIoU)

基于 statis_ablation_results.py 修改，将 Overall/Known/Unknown 的结果合并在一张表中展示，
且仅保留 mIoU 指标。

Usage:
    python statis_results_miou.py
    python statis_results_miou.py --dataset orfd
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 导入配置和辅助函数 (假设它们在 statis_ablation_results.py 中定义且可导入)
# 如果不想依赖导入，可以将那些函数复制过来。这里为了代码复用，尝试导入。
try:
    from statis_ablation_results import (
        ABLATION_EXPERIMENTS, 
        DATASET_UNKNOWN_SCENES,
        find_csv_for_experiment, 
        analyze_csv
    )
except ImportError:
    print("错误: 无法导入 statis_ablation_results.py。请确保该文件在当前目录下。")
    exit(1)

def main():
    parser = argparse.ArgumentParser(description='消融实验结果统计 (仅 mIoU)')
    parser.add_argument('--work-dirs', type=str, default='./work_dirs', help='工作目录')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['road3d', 'orfd', 'both'],
                        help='指定数据集')
    args = parser.parse_args()
    
    if args.dataset is None:
        args.dataset = 'both'
    
    datasets = ['orfd', 'road3d'] if args.dataset == 'both' else [args.dataset]
    
    for ds in datasets:
        process_dataset_miou(ds, args)

def process_dataset_miou(dataset_name, args):
    csv_key = f'csv_{dataset_name}'
    dataset_display = dataset_name.upper()
    
    print("\n" + "=" * 160)
    print(f"消融实验 mIoU 汇总表 - {dataset_display}")
    print("=" * 160)
    
    rows = []
    
    for exp_id, exp_info in ABLATION_EXPERIMENTS.items():
        # 获取 CSV 路径
        csv_path = exp_info.get(csv_key)
        if not csv_path:
            csv_path = find_csv_for_experiment(exp_info['name'], args.work_dirs)
        
        # 照搬 statis_ablation_results.py 的配置列逻辑
        sne_mode = exp_info.get('sne_mode', '-') if exp_info['sne'] else '-'
        sne_fusion_stage = exp_info.get('sne_fusion_stage', '-') if exp_info['sne'] else '-'
        ot_prior_raw = exp_info.get('ot_prior', None)
        if sne_mode == 'ot':
            ot_prior = '✓' if ot_prior_raw is True else ('✗' if ot_prior_raw is False else '-')
        else:
            ot_prior = '-'

        # OT 融合模式与 soft-union 统计
        if exp_info['sne'] and sne_mode == 'ot':
            fuse_mode_raw = exp_info.get('ot_fuse_mode')
            ot_fuse_mode = fuse_mode_raw if fuse_mode_raw else '-'
            ot_softunion = '✓' if exp_info.get('ot_softunion') is True else ('✗' if exp_info.get('ot_softunion') is False else '-')
        else:
            ot_fuse_mode = '-'
            ot_softunion = '-'

        base_row = {
            'Exp': exp_id,
            'Size': exp_info['size'],
            'Weight': exp_info['weight'],
            'SNE': '✓' if exp_info['sne'] else '✗',
            'SNE_Mode': sne_mode,
            'SNE_FusionStage': sne_fusion_stage,
            'OT_Prior': ot_prior,
            'OT_PriorMode': exp_info.get('ot_prior_mode', '-') if exp_info.get('ot_prior') else '-',
            'OT_Cost': exp_info.get('ot_cost_type', '-'),
            'OT_FuseOut': '✓' if exp_info.get('ot_fuse_output') is True else ('✗' if exp_info.get('ot_fuse_output') is False else '-'),
            'OT_FuseMode': ot_fuse_mode,
            'OT_SoftUnion': ot_softunion,
            'Prompt': '✓' if exp_info['prompt'] else '✗',
            'ContextDec': '✓' if exp_info.get('context_decoder', True) else '✗',
            'PatchFPN': '✓' if exp_info.get('patch_fpn') else '✗',
            'PatchFPN_XSAM': '✓' if exp_info.get('patch_fpn_xsam') else '✗',
            'PiSup': '✓' if exp_info.get('pi_sup') else '✗',
        }
            
        # 初始化指标
        base_row['mIoU(All)'] = '-'
        base_row['mIoU(Kn)'] = '-'
        base_row['mIoU(Unk)'] = '-'
        
        if csv_path and Path(csv_path).exists():
            try:
                metrics = analyze_csv(Path(csv_path), dataset_name)
                
                # Overall
                if 'overall' in metrics:
                    base_row['mIoU(All)'] = f"{metrics['overall']['mIoU']*100:.2f}"
                
                # Known/Unknown
                if 'known' in metrics:
                    base_row['mIoU(Kn)'] = f"{metrics['known']['mIoU']*100:.2f}"
                    base_row['mIoU(Unk)'] = f"{metrics['unknown']['mIoU']*100:.2f}"
            except Exception as e:
                pass
        
        rows.append(base_row)
    
    # 转为 DataFrame 并排序
    df = pd.DataFrame(rows)
    
    # 辅助排序列
    def sort_val(x):
        try:
            return float(x)
        except:
            return -1.0
            
    df['_sort'] = df['mIoU(All)'].apply(sort_val)
    df = df.sort_values('_sort', ascending=False).drop(columns=['_sort'])
    
    # 打印时不指定 index=False 以便如果需要可以加上，但通常不加更干净
    # 确保列顺序与 statis_ablation_results.py 一致，最后加上 mIoU 列
    columns_order = [
        'Exp', 'Size', 'Weight', 'SNE', 'SNE_Mode', 'SNE_FusionStage', 
        'OT_Prior', 'OT_PriorMode', 'OT_Cost', 'OT_FuseOut', 'OT_FuseMode', 'OT_SoftUnion',
        'Prompt', 'ContextDec', 'PatchFPN', 'PatchFPN_XSAM', 'PiSup',
        'mIoU(All)', 'mIoU(Kn)', 'mIoU(Unk)'
    ]
    # 过滤掉不在 dataframe 中的列（以防万一）
    columns_order = [c for c in columns_order if c in df.columns]
    
    print(df[columns_order].to_string(index=False))
    
    # 保存
    out_file = Path(args.work_dirs) / f'ablation_miou_summary_{dataset_name}.csv'
    df[columns_order].to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")

if __name__ == '__main__':
    main()
