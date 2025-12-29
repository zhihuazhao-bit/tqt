#!/usr/bin/env python3
"""
预训练消融实验结果统计脚本（简化版）

输出格式：Overall(mAcc, mRec, mF1, mIoU), Known(mAcc, mRec, mF1, mIoU), Unknown(mAcc, mRec, mF1, mIoU), delta_mIoU(Known - Unknown)

Usage:
    python statis_ablation_pretrain.py                    # 默认输出所有数据集
    python statis_ablation_pretrain.py --dataset orfd     # 仅输出 ORFD
    python statis_ablation_pretrain.py --dataset road3d   # 仅输出 Road3D
    python statis_ablation_pretrain.py --dataset orfd2road # 仅输出 ORFD2Road
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from utils.scene_config import DATASET_UNKNOWN_SCENES


def per_class_metrics_from_conf(conf):
    """从混淆矩阵计算每类指标"""
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)
    gt_sum = conf.sum(1).astype(float)
    union = gt_sum + pred_sum - tp

    prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum > 0)
    rec = np.divide(tp, gt_sum, out=np.zeros_like(tp), where=gt_sum > 0)
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    f1 = np.divide(2 * prec * rec, (prec + rec), out=np.zeros_like(tp), where=(prec + rec) > 0)
    return prec, rec, f1, iou


def macro_averages(prec, rec, f1, iou):
    """计算宏平均指标"""
    mprec = np.nanmean(prec)
    mrecall = np.nanmean(rec)
    mf1 = np.nanmean(f1)
    miou = np.nanmean(iou)
    return mprec, mrecall, mf1, miou


def getScores_self(conf_matrix):
    """从混淆矩阵计算所有指标"""
    prec, rec, f1, iou = per_class_metrics_from_conf(conf_matrix)
    mprec, mrecall, mf1, miou = macro_averages(prec, rec, f1, iou)
    return mprec, mrecall, mf1, miou


# 消融实验配置（按指定顺序）
ABLATION_EXPERIMENTS = [
    # 1. LearnableOnly-M2F-NoPretrain
    {
        'id': 'LearnableOnly-M2F-NoPretrain',
        'name': 'ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain',
        'desc': 'LearnableOnly + M2F + NoPretrain',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain/20251225_1726/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain/test_results/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain/testing_eval_file_stats_20251225_230751.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/20251225_1726/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/test_results/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/testing_eval_file_stats_20251225_235240.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_learnable_only_m2f_decoder_no_pretrain/20251225_1726/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain/test_results/exp_512_eva02_learnable_only_m2f_decoder_no_pretrain_road/testing_eval_file_stats_20251227_113246.csv',
    },
    # 2. P1-NoPretrain
    {
        'id': 'P1-NoPretrain',
        'name': 'ablation_512_eva02_sneProj_m2f_no_pretrain',
        'desc': 'SNE(proj) + M2F + NoPretrain',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_no_pretrain/20251226_1231/exp_512_eva02_sneProj_m2f_no_pretrain/test_results/exp_512_eva02_sneProj_m2f_no_pretrain/testing_eval_file_stats_20251227_074132.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_no_pretrain_road/20251226_1231/exp_512_eva02_sneProj_m2f_no_pretrain_road/test_results/exp_512_eva02_sneProj_m2f_no_pretrain_road/testing_eval_file_stats_20251227_084039.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_no_pretrain/20251226_1231/exp_512_eva02_sneProj_m2f_no_pretrain/test_results/exp_512_eva02_sneProj_m2f_no_pretrain_road/testing_eval_file_stats_20251227_122219.csv',
    },
    # 3. P2-TextPretrain
    {
        'id': 'P2-TextPretrain',
        'name': 'ablation_512_eva02_sneProj_m2f_text_pretrain',
        'desc': 'SNE(proj) + M2F + TextPretrain',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain/20251226_1236/exp_512_eva02_sneProj_m2f_text_pretrain/test_results/exp_512_eva02_sneProj_m2f_text_pretrain/testing_eval_file_stats_20251227_075401.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_road/20251226_1236/exp_512_eva02_sneProj_m2f_text_pretrain_road/test_results/exp_512_eva02_sneProj_m2f_text_pretrain_road/testing_eval_file_stats_20251227_093226.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain/20251226_1236/exp_512_eva02_sneProj_m2f_text_pretrain/test_results/exp_512_eva02_sneProj_m2f_text_pretrain_road/testing_eval_file_stats_20251227_131604.csv',
    },
    # 4. P7-SNEOT-TextPretrain-NoPromptCls
    {
        'id': 'P7-SNEOT-TextPretrain-NoPromptCls',
        'name': 'ablation_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls',
        'desc': 'SNE(OT) + PatchFPN + TextPretrain + NoPromptCls',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls/testing_eval_file_stats_20251227_084903.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/testing_eval_file_stats_20251227_140734.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain_no_promptcls_road/testing_eval_file_stats_20251227_175042.csv',
    },
    # 5. P5-TextPretrain-PromptCls
    {
        'id': 'P5-TextPretrain-PromptCls',
        'name': 'ablation_512_eva02_sneProj_m2f_text_pretrain_promptcls',
        'desc': 'SNE(proj) + M2F + TextPretrain + PromptCls',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_promptcls/20251226_1240/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls/test_results/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls/testing_eval_file_stats_20251227_082559.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/20251226_1240/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/test_results/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/testing_eval_file_stats_20251227_120949.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_text_pretrain_promptcls/20251226_1240/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls/test_results/exp_512_eva02_sneProj_m2f_text_pretrain_promptcls_road/testing_eval_file_stats_20251227_155918.csv',
    },
    # 6. P3-FullPretrain
    {
        'id': 'P3-FullPretrain',
        'name': 'ablation_512_eva02_sneProj_m2f_full_pretrain',
        'desc': 'SNE(proj) + M2F + FullPretrain',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain/test_results/exp_512_eva02_sneProj_m2f_full_pretrain/testing_eval_file_stats_20251227_080433.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_road/20251226_1238/exp_512_eva02_sneProj_m2f_full_pretrain_road/test_results/exp_512_eva02_sneProj_m2f_full_pretrain_road/testing_eval_file_stats_20251227_102151.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain/test_results/exp_512_eva02_sneProj_m2f_full_pretrain_road/testing_eval_file_stats_20251227_141026.csv',
    },
    # 7. P8-SNEOT-FullPretrain-NoPromptCls
    {
        'id': 'P8-SNEOT-FullPretrain-NoPromptCls',
        'name': 'ablation_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls',
        'desc': 'SNE(OT) + PatchFPN + FullPretrain + NoPromptCls',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls/20251226_1248/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls/test_results/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls/testing_eval_file_stats_20251227_090053.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/20251226_1704/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/test_results/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/testing_eval_file_stats_20251227_150443.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls/20251226_1248/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls/test_results/exp_512_eva02_sneot_patchfpn_full_pretrain_no_promptcls_road/testing_eval_file_stats_20251227_184652.csv',
    },
    # 8. P4-FullPretrain-PromptCls
    {
        'id': 'P4-FullPretrain-PromptCls',
        'name': 'ablation_512_eva02_sneProj_m2f_full_pretrain_promptcls',
        'desc': 'SNE(proj) + M2F + FullPretrain + PromptCls',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_promptcls/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls/test_results/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls/testing_eval_file_stats_20251227_081504.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/test_results/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/testing_eval_file_stats_20251227_111453.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneProj_m2f_full_pretrain_promptcls/20251226_1240/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls/test_results/exp_512_eva02_sneProj_m2f_full_pretrain_promptcls_road/testing_eval_file_stats_20251227_150629.csv',
    },
    # 9. P6-SNEOT-TextPretrain
    {
        'id': 'P6-SNEOT-TextPretrain',
        'name': 'ablation_512_eva02_sneot_patchfpn_text_pretrain',
        'desc': 'SNE(OT) + PatchFPN + TextPretrain + PromptCls',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain/testing_eval_file_stats_20251227_083747.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain_road/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain_road/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain_road/testing_eval_file_stats_20251227_130938.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneot_patchfpn_text_pretrain/20251226_1248/exp_512_eva02_sneot_patchfpn_text_pretrain/test_results/exp_512_eva02_sneot_patchfpn_text_pretrain_road/testing_eval_file_stats_20251227_165541.csv',
    },
    # 10. F2pSoft-learnableT-promptTau-0.1 (Ours)
    {
        'id': 'Ours',
        'name': 'ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau',
        'desc': 'SNE(OT) + PatchFPN + piSup + PromptSoft + LearnableT',
        'csv_orfd': '/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/20251219_1354/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/testing_eval_file_stats_20251220_085925.csv',
        'csv_road3d': '/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1320/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/testing_eval_file_stats_20251219_101507.csv',
        'csv_orfd2road': '/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/20251219_1354/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/testing_eval_file_stats_20251227_194202.csv',
    },
]

# 支持的数据集列表
ALL_DATASETS = ['orfd', 'road3d', 'orfd2road']


def analyze_csv(csv_path, dataset_name):
    """分析 CSV 文件，返回 known/unknown 指标"""
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

    # 计算指标
    def compute_metrics(mat):
        mAcc, mRecall, mF1, mIoU = getScores_self(mat)
        return {
            'mAcc': mAcc,
            'mRec': mRecall,
            'mF1': mF1,
            'mIoU': mIoU,
        }

    # Known/Unknown 场景分割
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, set())
    unknown_mask = df['scene'].isin(unknown_scenes)

    known_mat = subset_matrix(df[~unknown_mask])
    unknown_mat = subset_matrix(df[unknown_mask])
    overall_mat = subset_matrix(df)  # 整体混淆矩阵

    metrics = {
        'overall': compute_metrics(overall_mat),
        'known': compute_metrics(known_mat),
        'unknown': compute_metrics(unknown_mat),
    }

    return metrics


def process_dataset(dataset_name: str, output_path: str = None, verbose: bool = True) -> pd.DataFrame:
    """处理单个数据集的统计

    Args:
        dataset_name: 数据集名称 (orfd, road3d, orfd2road)
        output_path: 输出 CSV 文件路径
        verbose: 是否打印详细信息

    Returns:
        结果 DataFrame
    """
    csv_key = f'csv_{dataset_name}'
    dataset_display = dataset_name.upper()

    if verbose:
        print("\n" + "=" * 120)
        print(f"预训练消融实验结果统计 - {dataset_display}")
        print("=" * 120)

    results = []

    # 收集各实验结果
    for idx, exp_info in enumerate(ABLATION_EXPERIMENTS, 1):
        exp_id = exp_info['id']
        csv_path = exp_info.get(csv_key)

        if not csv_path or not Path(csv_path).exists():
            if verbose:
                print(f"[{idx:2d}] {exp_id}: 未找到结果文件")
            results.append({
                'No': idx,
                'Exp': exp_id,
                'Desc': exp_info['desc'],
                'O_mAcc': '-', 'O_mRec': '-', 'O_mF1': '-', 'O_mIoU': '-',
                'K_mAcc': '-', 'K_mRec': '-', 'K_mF1': '-', 'K_mIoU': '-',
                'U_mAcc': '-', 'U_mRec': '-', 'U_mF1': '-', 'U_mIoU': '-',
                'delta_mIoU': '-',
            })
            continue

        if verbose:
            print(f"[{idx:2d}] {exp_id}")

        try:
            metrics = analyze_csv(csv_path, dataset_name)
            m_overall = metrics['overall']
            m_known = metrics['known']
            m_unknown = metrics['unknown']

            delta_miou = m_known['mIoU'] - m_unknown['mIoU']

            results.append({
                'No': idx,
                'Exp': exp_id,
                'Desc': exp_info['desc'],
                'O_mAcc': f"{m_overall['mAcc'] * 100:.4f}",
                'O_mRec': f"{m_overall['mRec'] * 100:.4f}",
                'O_mF1': f"{m_overall['mF1'] * 100:.4f}",
                'O_mIoU': f"{m_overall['mIoU'] * 100:.4f}",
                'K_mAcc': f"{m_known['mAcc'] * 100:.4f}",
                'K_mRec': f"{m_known['mRec'] * 100:.4f}",
                'K_mF1': f"{m_known['mF1'] * 100:.4f}",
                'K_mIoU': f"{m_known['mIoU'] * 100:.4f}",
                'U_mAcc': f"{m_unknown['mAcc'] * 100:.4f}",
                'U_mRec': f"{m_unknown['mRec'] * 100:.4f}",
                'U_mF1': f"{m_unknown['mF1'] * 100:.4f}",
                'U_mIoU': f"{m_unknown['mIoU'] * 100:.4f}",
                'delta_mIoU': f"{delta_miou * 100:+.4f}",
            })

            if verbose:
                print(f"    Overall: mAcc={m_overall['mAcc']*100:.4f}, mRec={m_overall['mRec']*100:.4f}, "
                      f"mF1={m_overall['mF1']*100:.4f}, mIoU={m_overall['mIoU']*100:.4f}")
                print(f"    Known:   mAcc={m_known['mAcc']*100:.4f}, mRec={m_known['mRec']*100:.4f}, "
                      f"mF1={m_known['mF1']*100:.4f}, mIoU={m_known['mIoU']*100:.4f}")
                print(f"    Unknown: mAcc={m_unknown['mAcc']*100:.4f}, mRec={m_unknown['mRec']*100:.4f}, "
                      f"mF1={m_unknown['mF1']*100:.4f}, mIoU={m_unknown['mIoU']*100:.4f}")
                print(f"    delta_mIoU: {delta_miou*100:+.4f}")

        except Exception as e:
            if verbose:
                print(f"    分析失败: {e}")
            results.append({
                'No': idx,
                'Exp': exp_id,
                'Desc': exp_info['desc'],
                'O_mAcc': 'ERR', 'O_mRec': 'ERR', 'O_mF1': 'ERR', 'O_mIoU': 'ERR',
                'K_mAcc': 'ERR', 'K_mRec': 'ERR', 'K_mF1': 'ERR', 'K_mIoU': 'ERR',
                'U_mAcc': 'ERR', 'U_mRec': 'ERR', 'U_mF1': 'ERR', 'U_mIoU': 'ERR',
                'delta_mIoU': 'ERR',
            })

    # 创建 DataFrame
    df_results = pd.DataFrame(results)

    # 打印表格
    if verbose:
        print("\n" + "=" * 120)
        print(f"汇总表格 - {dataset_display}")
        print("=" * 120)
        print(df_results.to_string(index=False))

    # 保存 CSV
    if output_path is None:
        output_path = f'./work_dirs/pretrain_ablation_{dataset_name}.csv'
    df_results.to_csv(output_path, index=False)
    if verbose:
        print(f"\n结果已保存到: {output_path}")

    # 打印 LaTeX 表格格式
    if verbose:
        print("\n" + "=" * 120)
        print(f"LaTeX 表格格式 - {dataset_display}:")
        print("=" * 120)
        print("\\begin{tabular}{l|cccc|cccc|cccc|c}")
        print("\\toprule")
        print("Method & \\multicolumn{4}{c|}{Overall} & \\multicolumn{4}{c|}{Known} & \\multicolumn{4}{c|}{Unknown} & $\\Delta$ \\\\")
        print("       & mAcc & mRec & mF1 & mIoU & mAcc & mRec & mF1 & mIoU & mAcc & mRec & mF1 & mIoU & mIoU \\\\")
        print("\\midrule")
        for r in results:
            if r['O_mAcc'] == '-' or r['O_mAcc'] == 'ERR':
                continue
            print(f"{r['Exp']} & {r['O_mAcc']} & {r['O_mRec']} & {r['O_mF1']} & {r['O_mIoU']} "
                  f"& {r['K_mAcc']} & {r['K_mRec']} & {r['K_mF1']} & {r['K_mIoU']} "
                  f"& {r['U_mAcc']} & {r['U_mRec']} & {r['U_mF1']} & {r['U_mIoU']} & {r['delta_mIoU']} \\\\")
        print("\\bottomrule")
        print("\\end{tabular}")

    return df_results


def main():
    parser = argparse.ArgumentParser(description='预训练消融实验结果统计')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'orfd', 'road3d', 'orfd2road'],
                        help='指定数据集 (default: all)')
    parser.add_argument('--output-dir', type=str, default='./work_dirs',
                        help='输出目录 (default: ./work_dirs)')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式，减少输出')
    args = parser.parse_args()

    # 确定要处理的数据集列表
    if args.dataset == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [args.dataset]

    print("\n" + "#" * 120)
    print(f"# 预训练消融实验结果统计")
    print(f"# 数据集: {', '.join(datasets)}")
    print("#" * 120)

    # 处理每个数据集
    all_results = {}
    for dataset_name in datasets:
        output_path = f'{args.output_dir}/pretrain_ablation_{dataset_name}.csv'
        df = process_dataset(dataset_name, output_path, verbose=not args.quiet)
        all_results[dataset_name] = df

    # 打印汇总
    print("\n" + "#" * 120)
    print("# 所有数据集处理完成!")
    print("#" * 120)
    for dataset_name in datasets:
        output_path = f'{args.output_dir}/pretrain_ablation_{dataset_name}.csv'
        print(f"  - {dataset_name.upper()}: {output_path}")


if __name__ == '__main__':
    main()
