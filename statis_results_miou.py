#!/usr/bin/env python3
"""
消融实验结果统计脚本 (仅 mIoU)

基于 statis_ablation_results.py 修改，将 Overall/Known/Unknown 的结果合并在一张表中展示，
且仅保留 mIoU 指标。

Usage:
    python statis_results_miou.py
    python statis_results_miou.py --dataset orfd
    python statis_results_miou.py --per-scene   # 显示每个unknown场景的详细指标
"""

import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd

# 导入配置和辅助函数
from utils.scene_config import DATASET_UNKNOWN_SCENES, ABNORMAL_SCENES

try:
    from statis_ablation_results import (
        ABLATION_EXPERIMENTS,
        find_csv_for_experiment,
        analyze_csv,
        getScores_self
    )
except ImportError:
    print("错误: 无法导入 statis_ablation_results.py。请确保该文件在当前目录下。")
    exit(1)

def analyze_csv_per_scene(csv_path, dataset_name):
    """分析 CSV 文件，返回每个场景的单独指标"""
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'path'})

    # 检测 CSV 格式
    conf_cols = [c for c in df.columns if '-' in c and all(part.isdigit() for part in c.split('-'))]
    class_summary_cols = [c for c in df.columns if c.startswith('class') and c.endswith('_intersect')]

    if conf_cols:
        class_ids = sorted({int(part) for col in conf_cols for part in col.split('-')})
        num_classes = max(class_ids) + 1

        def subset_matrix(sub_df):
            if sub_df.empty:
                return np.zeros((num_classes, num_classes), dtype=np.float64)
            mat = np.zeros((num_classes, num_classes), dtype=np.float64)
            series = sub_df[conf_cols].sum()
            for key, value in series.items():
                i, j = map(int, key.split('-'))
                mat[i, j] = value
            return mat
    elif class_summary_cols:
        df['scene'] = df['scene'].apply(lambda x: x[1:].replace("_", "-") if isinstance(x, str) else x)

        def subset_matrix(sub_df):
            if sub_df.empty:
                return np.zeros((2, 2), dtype=np.float64)
            tp_pos = sub_df['class1_intersect'].sum()
            pred_pos = sub_df['class1_pred_label'].sum()
            label_pos = sub_df['class1_label'].sum()
            tn = sub_df['class0_intersect'].sum()
            fp_pos = pred_pos - tp_pos
            fn_pos = label_pos - tp_pos
            mat = np.array([
                [tn, fp_pos],
                [fn_pos, tp_pos],
            ], dtype=np.float64)
            return mat
    else:
        raise ValueError(f"无法识别 CSV 格式: {csv_path}")

    def compute_metrics(mat):
        mAcc, mRecall, mF1, mIoU, fwIoU, prec_road, rec_road, f1_road, iou_road = getScores_self(mat)
        return {
            'mAcc': mAcc,
            'mRecall': mRecall,
            'mF1': mF1,
            'mIoU': mIoU,
            'fwIoU': fwIoU,
            'prec_trav': prec_road,
            'rec_trav': rec_road,
            'f1_trav': f1_road,
            'iou_trav': iou_road,
        }

    # 获取所有场景
    all_scenes = df['scene'].unique()
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])

    per_scene_metrics = {}
    for scene in all_scenes:
        scene_df = df[df['scene'] == scene]
        mat = subset_matrix(scene_df)
        is_unknown = scene in unknown_scenes
        per_scene_metrics[scene] = {
            'metrics': compute_metrics(mat),
            'is_unknown': is_unknown,
            'num_samples': len(scene_df)
        }

    return per_scene_metrics


def main():
    parser = argparse.ArgumentParser(description='消融实验结果统计 (仅 mIoU)')
    parser.add_argument('--work-dirs', type=str, default='./work_dirs', help='工作目录')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['road3d', 'orfd', 'both'],
                        help='指定数据集')
    parser.add_argument('--per-scene', action='store_true',
                        help='显示每个unknown场景的详细指标')
    args = parser.parse_args()

    if args.dataset is None:
        args.dataset = 'both'

    datasets = ['orfd', 'road3d'] if args.dataset == 'both' else [args.dataset]

    for ds in datasets:
        if args.per_scene:
            process_dataset_per_scene(ds, args)
        else:
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


def process_dataset_per_scene(dataset_name, args):
    """处理数据集并显示每个unknown场景的详细指标"""
    csv_key = f'csv_{dataset_name}'
    dataset_display = dataset_name.upper()
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])

    print("\n" + "=" * 180)
    print(f"消融实验 Unknown 场景详细指标 - {dataset_display}")
    print(f"Unknown 场景列表: {unknown_scenes}")
    print("=" * 180)

    rows = []

    for exp_id, exp_info in ABLATION_EXPERIMENTS.items():
        # 获取 CSV 路径
        csv_path = exp_info.get(csv_key)
        if not csv_path:
            csv_path = find_csv_for_experiment(exp_info['name'], args.work_dirs)

        base_row = {
            'Exp': exp_id,
            'Name': exp_info['name'][:40],  # 截断名称
        }

        if csv_path and Path(csv_path).exists():
            try:
                per_scene = analyze_csv_per_scene(Path(csv_path), dataset_name)
                metrics = analyze_csv(Path(csv_path), dataset_name)

                # Overall 和汇总指标
                if 'overall' in metrics:
                    base_row['mIoU(All)'] = f"{metrics['overall']['mIoU']*100:.2f}"
                if 'known' in metrics:
                    base_row['mIoU(Kn)'] = f"{metrics['known']['mIoU']*100:.2f}"
                if 'unknown' in metrics:
                    base_row['mIoU(Unk)'] = f"{metrics['unknown']['mIoU']*100:.2f}"

                # 每个 unknown 场景的 mIoU
                for scene in unknown_scenes:
                    if scene in per_scene:
                        scene_metrics = per_scene[scene]['metrics']
                        base_row[f'{scene}'] = f"{scene_metrics['mIoU']*100:.2f}"
                    else:
                        base_row[f'{scene}'] = '-'

            except Exception as e:
                base_row['mIoU(All)'] = f'Error: {str(e)[:20]}'
        else:
            base_row['mIoU(All)'] = 'CSV not found'

        rows.append(base_row)

    # 转为 DataFrame 并排序
    df = pd.DataFrame(rows)

    # 辅助排序列
    def sort_val(x):
        try:
            return float(x)
        except:
            return -1.0

    if 'mIoU(Unk)' in df.columns:
        df['_sort'] = df['mIoU(Unk)'].apply(sort_val)
        df = df.sort_values('_sort', ascending=False).drop(columns=['_sort'])

    # 构建列顺序
    columns_order = ['Exp', 'Name', 'mIoU(All)', 'mIoU(Kn)', 'mIoU(Unk)'] + unknown_scenes

    columns_order = [c for c in columns_order if c in df.columns]

    # 打印表格
    print(df[columns_order].to_string(index=False))

    # 打印每个场景的对比排名
    print("\n" + "-" * 100)
    print("各 Unknown 场景 mIoU 排名")
    print("-" * 100)

    for scene in unknown_scenes:
        if scene in df.columns:
            scene_df = df[['Exp', scene]].copy()
            scene_df['_val'] = scene_df[scene].apply(sort_val)
            scene_df = scene_df.sort_values('_val', ascending=False).drop(columns=['_val'])
            print(f"\n场景 [{scene}]:")
            for idx, row in scene_df.iterrows():
                print(f"  {row['Exp']:10s} -> {row[scene]}")

    # 保存
    out_file = Path(args.work_dirs) / f'ablation_unknown_per_scene_{dataset_name}.csv'
    df[columns_order].to_csv(out_file, index=False)
    print(f"\nSaved to: {out_file}")


if __name__ == '__main__':
    main()
