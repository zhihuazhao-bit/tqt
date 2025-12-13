#!/usr/bin/env python3
"""
消融实验结果统计脚本

自动收集所有消融实验的测试结果并生成对比表格。
支持区分 known/unknown 场景指标（参考 statis_results.py）。

Usage:
    python statis_ablation_results.py
    python statis_ablation_results.py --csv-list ./work_dirs/ablation_csv_list.txt
    python statis_ablation_results.py --dataset orfd
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


def per_class_metrics_from_conf(conf):
    # 每类 precision/recall/IoU
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)  # 列和（被预测为该类）
    gt_sum   = conf.sum(1).astype(float)  # 行和（真实属于该类）
    union    = gt_sum + pred_sum - tp

    prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum>0)
    rec  = np.divide(tp, gt_sum,   out=np.zeros_like(tp), where=gt_sum>0)
    iou  = np.divide(tp, union,    out=np.zeros_like(tp), where=union>0)
    f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(tp), where=(prec+rec)>0)
    return prec, rec, f1, iou

def macro_averages(prec, rec, f1, iou):
    mprec   = np.nanmean(prec)
    mrecall = np.nanmean(rec)
    mf1     = np.nanmean(f1)      # 宏F1建议按“每类F1再平均”
    miou    = np.nanmean(iou)
    return mprec, mrecall, mf1, miou

def freq_weighted_iou(conf):
    freq = conf.sum(1)
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)
    gt_sum   = conf.sum(1).astype(float)
    union    = gt_sum + pred_sum - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union>0)
    fw_iou = np.sum(freq * iou) / np.sum(freq)
    return fw_iou

def getScores_self(conf_matrix):
    # 使用
    prec, rec, f1, iou = per_class_metrics_from_conf(conf_matrix)
    # 正类（假设索引1是road）
    prec_road, rec_road, f1_road, iou_road = prec[1], rec[1], f1[1], iou[1]
    # 宏/加权
    mprec, mrecall, mf1, miou = macro_averages(prec, rec, f1, iou)
    fwIoU = freq_weighted_iou(conf_matrix)

    return mprec, mrecall, mf1, miou, fwIoU, prec_road, rec_road, f1_road, iou_road


# 数据集 unknown 场景配置
DATASET_UNKNOWN_SCENES = {
    'road3d': ['2021-0403-1744', '0602-1107', '2021-0222-1743', '2021-0403-1858'],
    'orfd': ['0609-1923', '2021-0223-1756'],
}


# 消融实验配置 - 手动指定 CSV 路径
# 将 'csv_orfd' / 'csv_road3d' 字段设置为实际的 CSV 文件路径，留空或 None 则自动搜索
ABLATION_EXPERIMENTS = {
    'A1': {
        'name': 'ablation_224_eva02_nosne_noprompt',
        'desc': '224 + EVA02 + NoSNE + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': False,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_190458.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_road/20251203_2312/exp_224_eva02_nosne_noprompt_road/test_results/exp_224_eva02_nosne_noprompt_road/testing_eval_file_stats_20251204_101553.csv',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'A1-nc': {
        'name': 'ablation_224_eva02_nosne_noprompt_nocd',
        'desc': '224 + EVA02 + NoSNE + NoPrompt + NoContextDecoder',
        'size': 224,
        'weight': 'EVA02',
        'sne': False,
        'prompt': False,
        'context_decoder': False,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_nosne_noprompt_nocd/20251212_1823/exp_224_eva02_nosne_noprompt_nocd/test_results/exp_224_eva02_nosne_noprompt_nocd/testing_eval_file_stats_20251212_190055.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'A2': {
        'name': 'ablation_512_eva02_nosne_noprompt',
        'desc': '512 + EVA02 + NoSNE + NoPrompt',
        'size': 512,
        'weight': 'EVA02',
        'sne': False,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_191028.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_nosne_noprompt_road/20251203_2313/exp_512_eva02_nosne_noprompt_road/test_results/exp_512_eva02_nosne_noprompt_road/testing_eval_file_stats_20251204_104203.csv',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F2a': {
        'name': 'ablation_224_eva02_sneotFalse_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=F) + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': False,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneot_noprompt/20251204_1638/exp_224_eva02_sneot_noprompt/test_results/exp_224_eva02_sneotFalse_noprompt/testing_eval_file_stats_20251204_193006.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F2b': {
        'name': 'ablation_224_eva02_sneotTrue_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_noprompt/20251204_1727/exp_224_eva02_sneotTrue_noprompt/test_results/exp_224_eva02_sneotTrue_noprompt/testing_eval_file_stats_20251204_193627.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F2c': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_noprompt/20251207_1303/exp_224_eva02_sneotTrue_patchfpn_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_noprompt/testing_eval_file_stats_20251207_140213.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F2d': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN + piSup + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': True,
        'pi_sup': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt_old/20251208_0812/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/testing_eval_file_stats_20251212_175550.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
     'F2d1': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN + piSup + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': True,
        'pi_sup': True,
        'ot_cost_type': 'cos',
        'ot_fuse_output': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt/20251212_2239/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/testing_eval_file_stats_20251213_095931.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
     'F2d2': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN + piSup + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': True,
        'pi_sup': True,
        'ot_cost_type': 'l2',
        'ot_fuse_output': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-l2/20251212_2322/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_L2/testing_eval_file_stats_20251213_100538.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
     'F2d3': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN + piSup + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': True,
        'pi_sup': True,
        'ot_cost_type': 'l2',
        'ot_fuse_output': False,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_pisup_noprompt-no-l2/20251212_2257/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_pisup_noprompt_No_L2/testing_eval_file_stats_20251213_101152.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F2d-xsam': {
        'name': 'ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt',
        'desc': '224 + EVA02 + SNE(OT,prior=T) + Patch-FPN(PixelSampling) + piSup + NoPrompt',
        'size': 224,
        'weight': 'EVA02',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'patch_fpn': False,
        'patch_fpn_xsam': True,
        'pi_sup': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/20251212_1850/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/test_results/exp_224_eva02_sneotTrue_patchfpn_xsam_pisup_noprompt/testing_eval_file_stats_20251212_201334.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'B2': {
        'name': 'ablation_224_densevlm_nosne_noprompt',
        'desc': '224 + DenseVLM + NoSNE + NoPrompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': False,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_191544.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_nosne_noprompt_road/20251203_2311/exp_224_densevlm_nosne_noprompt_road/test_results/exp_224_densevlm_nosne_noprompt_road/testing_eval_file_stats_20251204_110935.csv',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'C2': {
        'name': 'ablation_224_densevlm_sne_noprompt',
        'desc': '224 + DenseVLM + SNE + NoPrompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': True,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_192219.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_sne_noprompt_road/20251203_2311/exp_224_densevlm_sne_noprompt_road/test_results/exp_224_densevlm_sne_noprompt_road/testing_eval_file_stats_20251204_113833.csv',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'D2': {
        'name': 'ablation_224_densevlm_sne_prompt',
        'desc': '224 + DenseVLM + SNE + Prompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': True,
        'prompt': True,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_204424.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_224_densevlm_sne_prompt_road/20251203_2312/exp_224_densevlm_sne_prompt_road/test_results/exp_224_densevlm_sne_prompt_road/testing_eval_file_stats_20251204_135458.csv',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'E1': {
        'name': 'ablation_512_densevlm_sne_prompt',
        'desc': '512 + DenseVLM + SNE + Prompt',
        'size': 512,
        'weight': 'DenseVLM',
        'sne': True,
        'prompt': True,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/csv_result/testing_eval_file_stats_20251203_205051.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_densevlm_sne_prompt_road/20251203_2312/exp_512_densevlm_sne_prompt_road/test_results/exp_512_densevlm_sne_prompt_road/testing_eval_file_stats_20251204_142944.csv',
    },
    'E2': {
        'name': 'ablation_512_densevlm_nosne_prompt',
        'desc': '512 + DenseVLM + NoSNE + Prompt',
        'size': 512,
        'weight': 'DenseVLM',
        'sne': False,
        'prompt': True,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_densevlm_nosne_prompt/20251204_0815/exp_512_densevlm_nosne_prompt/test_results/exp_512_densevlm_nosne_prompt/testing_eval_file_stats_20251204_115858.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F1a': {
        'name': 'ablation_224_densevlm_sneotFalse_noprompt',
        'desc': '224 + DenseVLM + SNE(OT,prior=F) + NoPrompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': False,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneot_noprompt/20251204_1638/exp_224_densevlm_sneot_noprompt/test_results/exp_224_densevlm_sneotFalse_noprompt/testing_eval_file_stats_20251204_191728.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F1b': {
        'name': 'ablation_224_densevlm_sneotTrue_noprompt',
        'desc': '224 + DenseVLM + SNE(OT,prior=T) + NoPrompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_noprompt/20251204_1724/exp_224_densevlm_sneotTrue_noprompt/test_results/exp_224_densevlm_sneotTrue_noprompt/testing_eval_file_stats_20251204_192342.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    'F1b2': {
        'name': 'ablation_224_densevlm_sneotTrue_noprompt',
        'desc': '224 + DenseVLM + SNE(OT,prior=T) + NoPrompt',
        'size': 224,
        'weight': 'DenseVLM',
        'sne': True,
        'sne_mode': 'ot',
        'ot_prior': True,
        'prompt': False,
        'context_decoder': True,
        'csv_orfd': '/root/tqdm/work_dirs/ablation_224_densevlm_sneotTrue_noprompt/20251204_1724/exp_224_densevlm_sneotTrue_noprompt/test_results/exp_224_densevlm_sneotTrue_noprompt/testing_eval_file_stats_20251204_192342.csv',  # TODO: 填入 ORFD 测试结果 CSV 路径
        'csv_road3d': '',  # TODO: 填入 Road3D 测试结果 CSV 路径
    },
    
}


def find_csv_for_experiment(exp_name, work_dirs='./work_dirs'):
    """查找实验对应的 CSV 文件"""
    exp_dir = Path(work_dirs) / exp_name
    if not exp_dir.exists():
        return None
    
    # 查找最新的时间戳目录
    timestamp_dirs = sorted(exp_dir.glob('*/'), key=lambda x: x.name, reverse=True)
    for ts_dir in timestamp_dirs:
        # 查找 test_results 目录下的 CSV
        csv_files = list(ts_dir.glob('test_results/testing_eval_file_stats_*.csv'))
        if csv_files:
            return max(csv_files, key=lambda x: x.stat().st_mtime)
    
    return None


def analyze_csv(csv_path, dataset_name=None):
    """分析单个 CSV 文件，返回 overall/known/unknown 指标"""
    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'path'})
    
    # 检测 CSV 格式
    conf_cols = [c for c in df.columns if '-' in c and all(part.isdigit() for part in c.split('-'))]
    class_summary_cols = [c for c in df.columns if c.startswith('class') and c.endswith('_intersect')]
    
    if conf_cols:
        format_type = 'explicit_conf'
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
        format_type = 'per_class_summary'
        class_ids = sorted(int(col[len('class'):].split('_')[0]) for col in class_summary_cols)
        num_classes = len(class_ids)
        if num_classes != 2:
            raise NotImplementedError("per-class summary 格式当前仅支持二分类")
        
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
    
    # 自动检测数据集
    if dataset_name is None:
        csv_lower = str(csv_path).lower()
        for name in DATASET_UNKNOWN_SCENES:
            if name in csv_lower:
                dataset_name = name
                break
        # 从 scene 列检测
        if dataset_name is None and 'scene' in df.columns:
            scene_set = set(df['scene'])
            for name, scenes in DATASET_UNKNOWN_SCENES.items():
                if scene_set.intersection(scenes):
                    dataset_name = name
                    break
    
    # 计算各项指标
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
    
    # Overall 指标
    overall_mat = subset_matrix(df)
    metrics = {'overall': compute_metrics(overall_mat)}
    
    # Known/Unknown 场景分割
    if dataset_name and dataset_name in DATASET_UNKNOWN_SCENES:
        unknown_scenes = DATASET_UNKNOWN_SCENES[dataset_name]
        unknown_mask = df['scene'].isin(unknown_scenes)
        
        known_mat = subset_matrix(df[~unknown_mask])
        unknown_mat = subset_matrix(df[unknown_mask])
        
        metrics['known'] = compute_metrics(known_mat)
        metrics['unknown'] = compute_metrics(unknown_mat)
        metrics['dataset'] = dataset_name
        metrics['num_known_scenes'] = df[~unknown_mask]['scene'].nunique()
        metrics['num_unknown_scenes'] = df[unknown_mask]['scene'].nunique()
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='消融实验结果统计')
    parser.add_argument('--csv-list', type=str, default=None,
                        help='包含 CSV 路径列表的文件')
    parser.add_argument('--work-dirs', type=str, default='./work_dirs',
                        help='工作目录')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['road3d', 'orfd', 'both'],
                        help='指定数据集用于 known/unknown 场景分割，both 表示同时处理两个数据集')
    args = parser.parse_args()
    
    # 默认处理两个数据集
    if args.dataset is None:
        args.dataset = 'both'
    
    datasets_to_process = ['orfd', 'road3d'] if args.dataset == 'both' else [args.dataset]
    
    for dataset_name in datasets_to_process:
        process_dataset(dataset_name, args)


def process_dataset(dataset_name, args):
    """处理单个数据集的消融实验结果"""
    csv_key = f'csv_{dataset_name}'
    dataset_display = dataset_name.upper()
    
    print("\n" + "=" * 100)
    print(f"TQT 消融实验结果统计 - {dataset_display}")
    print("=" * 100)
    
    results_overall = []
    results_known = []
    results_unknown = []
    
    # 收集各实验结果
    for exp_id, exp_info in ABLATION_EXPERIMENTS.items():
        # 优先使用手动指定的 CSV 路径
        csv_path = exp_info.get(csv_key)
        if csv_path:
            csv_path = Path(csv_path)
            if not csv_path.exists():
                print(f"[{exp_id}] {exp_info['desc']}: 指定的 CSV 不存在: {csv_path}")
                csv_path = None
        else:
            # 自动搜索
            csv_path = find_csv_for_experiment(exp_info['name'], args.work_dirs)
        
        # 展示 SNE 融合方式与是否使用 OT prior（仅当 sne_mode 为 ot 时展示）
        sne_mode = exp_info.get('sne_mode', '-') if exp_info['sne'] else '-'
        ot_prior_raw = exp_info.get('ot_prior', None)
        if sne_mode == 'ot':
            ot_prior = '✓' if ot_prior_raw is True else ('✗' if ot_prior_raw is False else '-')
        else:
            ot_prior = '-'

        base_row = {
            'Exp': exp_id,
            'Size': exp_info['size'],
            'Weight': exp_info['weight'],
            'SNE': '✓' if exp_info['sne'] else '✗',
            'SNE_Mode': sne_mode,
            'OT_Prior': ot_prior,
            'OT_Cost': exp_info.get('ot_cost_type', '-'),
            'OT_FuseOut': '✓' if exp_info.get('ot_fuse_output') is True else ('✗' if exp_info.get('ot_fuse_output') is False else '-'),
            'Prompt': '✓' if exp_info['prompt'] else '✗',
            'ContextDec': '✓' if exp_info.get('context_decoder', True) else '✗',
            'PatchFPN': '✓' if exp_info.get('patch_fpn') else '✗',
            'PatchFPN_XSAM': '✓' if exp_info.get('patch_fpn_xsam') else '✗',
            'PiSup': '✓' if exp_info.get('pi_sup') else '✗',
        }
        
        if csv_path is None:
            print(f"[{exp_id}] {exp_info['desc']}: 未找到结果文件")
            empty_metrics = {'mIoU': '-', 'mF1': '-', 'fwIoU': '-', 'iou_trav': '-'}
            results_overall.append({**base_row, **empty_metrics})
            results_known.append({**base_row, **empty_metrics})
            results_unknown.append({**base_row, **empty_metrics})
            continue
        
        print(f"[{exp_id}] {exp_info['desc']}")
        print(f"    CSV: {csv_path}")
        
        try:
            metrics = analyze_csv(csv_path, dataset_name)
            
            # Overall 结果
            m_overall = metrics['overall']
            results_overall.append({
                **base_row,
                'mIoU': f"{m_overall['mIoU']:.4f}",
                'mF1': f"{m_overall['mF1']:.4f}",
                'fwIoU': f"{m_overall['fwIoU']:.4f}",
                'iou_trav': f"{m_overall['iou_trav']:.4f}",
            })
            
            print(f"    [overall] mIoU={m_overall['mIoU']:.4f}, mF1={m_overall['mF1']:.4f}, "
                  f"fwIoU={m_overall['fwIoU']:.4f}, iou_trav={m_overall['iou_trav']:.4f}")
            
            # Known/Unknown 结果
            if 'known' in metrics:
                m_known = metrics['known']
                m_unknown = metrics['unknown']
                
                results_known.append({
                    **base_row,
                    'mIoU': f"{m_known['mIoU']:.4f}",
                    'mF1': f"{m_known['mF1']:.4f}",
                    'fwIoU': f"{m_known['fwIoU']:.4f}",
                    'iou_trav': f"{m_known['iou_trav']:.4f}",
                })
                results_unknown.append({
                    **base_row,
                    'mIoU': f"{m_unknown['mIoU']:.4f}",
                    'mF1': f"{m_unknown['mF1']:.4f}",
                    'fwIoU': f"{m_unknown['fwIoU']:.4f}",
                    'iou_trav': f"{m_unknown['iou_trav']:.4f}",
                })
                
                print(f"    [known]   mIoU={m_known['mIoU']:.4f}, mF1={m_known['mF1']:.4f}, "
                      f"fwIoU={m_known['fwIoU']:.4f}, iou_trav={m_known['iou_trav']:.4f}")
                print(f"    [unknown] mIoU={m_unknown['mIoU']:.4f}, mF1={m_unknown['mF1']:.4f}, "
                      f"fwIoU={m_unknown['fwIoU']:.4f}, iou_trav={m_unknown['iou_trav']:.4f}")
            else:
                empty_metrics = {'mIoU': 'N/A', 'mF1': 'N/A', 'fwIoU': 'N/A', 'iou_trav': 'N/A'}
                results_known.append({**base_row, **empty_metrics})
                results_unknown.append({**base_row, **empty_metrics})
                print(f"    [known/unknown] 未检测到数据集，无法区分场景")
                
        except Exception as e:
            print(f"    错误: {e}")
            error_metrics = {'mIoU': 'Error', 'mF1': 'Error', 'fwIoU': 'Error', 'iou_trav': 'Error'}
            results_overall.append({**base_row, **error_metrics})
            results_known.append({**base_row, **error_metrics})
            results_unknown.append({**base_row, **error_metrics})
    
    # 打印 Overall 结果表格
    print("\n" + "=" * 100)
    print(f"消融实验结果对比表 - {dataset_display} - Overall")
    print("=" * 100)
    df_overall = pd.DataFrame(results_overall)
    print(df_overall.to_string(index=False))
    
    # 打印 Known 结果表格
    print("\n" + "=" * 100)
    print(f"消融实验结果对比表 - {dataset_display} - Known Scenes")
    print("=" * 100)
    df_known = pd.DataFrame(results_known)
    print(df_known.to_string(index=False))
    
    # 打印 Unknown 结果表格
    print("\n" + "=" * 100)
    print(f"消融实验结果对比表 - {dataset_display} - Unknown Scenes")
    print("=" * 100)
    df_unknown = pd.DataFrame(results_unknown)
    print(df_unknown.to_string(index=False))
    
    # 保存结果
    output_overall = Path(args.work_dirs) / f'ablation_results_{dataset_name}_overall.csv'
    output_known = Path(args.work_dirs) / f'ablation_results_{dataset_name}_known.csv'
    output_unknown = Path(args.work_dirs) / f'ablation_results_{dataset_name}_unknown.csv'
    
    df_overall.to_csv(output_overall, index=False)
    df_known.to_csv(output_known, index=False)
    df_unknown.to_csv(output_unknown, index=False)
    
    print(f"\n结果已保存至:")
    print(f"  Overall: {output_overall}")
    print(f"  Known:   {output_known}")
    print(f"  Unknown: {output_unknown}")
    
    # 对比分析 - 使用 overall 指标
    print("\n" + "=" * 100)
    print(f"对比分析 - {dataset_display} (基于 Overall 指标)")
    print("=" * 100)
    
    comparisons = [
        ('尺寸影响', 'A1', 'A2', '224 vs 512 (EVA02)'),
        ('预训练权重影响', 'A1', 'B2', 'EVA02 vs DenseVLM'),
        ('SNE 效果', 'B2', 'C2', 'NoSNE vs SNE(proj)'),
        ('SNE 融合方式 (DenseVLM)', 'C2', 'F1a', 'SNE(proj) vs SNE(OT,prior=F)'),
        ('OT Prior 效果 (DenseVLM)', 'F1a', 'F1b', 'OT(prior=F) vs OT(prior=T)'),
        ('SNE 融合方式 (EVA02)', 'A1', 'F2a', 'NoSNE vs SNE(OT,prior=F)'),
        ('OT Prior 效果 (EVA02)', 'F2a', 'F2b', 'OT(prior=F) vs OT(prior=T)'),
        ('Prompt 效果', 'C2', 'D2', 'NoPrompt vs Prompt'),
        ('完整对比', 'A1', 'E1', '最简基准 vs 完整模型'),
        ('Pi 监督效果 (EVA02 Patch-FPN)', 'F2c', 'F2d', 'OT prior=T, Patch-FPN, 无pi监督 vs 有pi监督'),
    ]
    
    for desc, exp1, exp2, detail in comparisons:
        r1 = next((r for r in results_overall if r['Exp'] == exp1), None)
        r2 = next((r for r in results_overall if r['Exp'] == exp2), None)
        
        if r1 and r2 and r1['mIoU'] not in ['-', 'Error'] and r2['mIoU'] not in ['-', 'Error']:
            try:
                diff = float(r2['mIoU']) - float(r1['mIoU'])
                sign = '+' if diff > 0 else ''
                print(f"  {desc} ({detail}):")
                print(f"    {exp1}: mIoU={r1['mIoU']} -> {exp2}: mIoU={r2['mIoU']} ({sign}{diff:.4f})")
            except:
                print(f"  {desc}: 数据不完整")
        else:
            print(f"  {desc}: 数据不完整")
    
    # 对比分析 - Unknown 场景 (重点关注泛化能力)
    print("\n" + "=" * 100)
    print(f"对比分析 - {dataset_display} (基于 Unknown 场景指标 - 泛化能力)")
    print("=" * 100)
    
    for desc, exp1, exp2, detail in comparisons:
        r1 = next((r for r in results_unknown if r['Exp'] == exp1), None)
        r2 = next((r for r in results_unknown if r['Exp'] == exp2), None)
        
        if r1 and r2 and r1['mIoU'] not in ['-', 'Error', 'N/A'] and r2['mIoU'] not in ['-', 'Error', 'N/A']:
            try:
                diff = float(r2['mIoU']) - float(r1['mIoU'])
                sign = '+' if diff > 0 else ''
                print(f"  {desc} ({detail}):")
                print(f"    {exp1}: mIoU={r1['mIoU']} -> {exp2}: mIoU={r2['mIoU']} ({sign}{diff:.4f})")
            except:
                print(f"  {desc}: 数据不完整")
        else:
            print(f"  {desc}: 数据不完整")
    
    print("=" * 100)


if __name__ == '__main__':
    main()
