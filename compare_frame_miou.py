#!/usr/bin/env python3
"""
统计未知场景中每一帧的 rod/m2f2/off-net 最大 mIoU 与 TQT 的差值
"""

import numpy as np
import pandas as pd
from pathlib import Path
from utils.scene_config import DATASET_UNKNOWN_SCENES


def per_class_metrics_from_conf(conf):
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)
    gt_sum = conf.sum(1).astype(float)
    union = gt_sum + pred_sum - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union > 0)
    return iou


def calculate_miou(conf_matrix):
    iou = per_class_metrics_from_conf(conf_matrix)
    return np.nanmean(iou)


def extract_scene_and_frame(path_str, csv_format):
    """从路径中提取场景名和帧名"""
    if not isinstance(path_str, str):
        return None, None

    # 处理不同格式的路径
    parts = path_str.split('/')

    if 'testing' in path_str:
        # 找到 testing 后的场景名
        for i, p in enumerate(parts):
            if p == 'testing' and i + 1 < len(parts):
                scene_folder = parts[i + 1]
                # 提取场景名（去掉前缀如 x, y）
                if scene_folder.startswith(('x', 'y')) and len(scene_folder) > 1:
                    scene = scene_folder[1:].replace('_', '-')
                else:
                    scene = scene_folder.replace('_', '-')

                # 提取帧名 - 处理不同格式
                filename = parts[-1]
                # 移除扩展名
                frame = filename.rsplit('.', 1)[0]
                # 如果有下划线，取第一部分（时间戳）
                if '_' in frame:
                    frame = frame.split('_')[0]
                return scene, frame

    return None, None


def extract_frame_from_path(path_str):
    """从路径中提取帧名（时间戳）"""
    if not isinstance(path_str, str):
        return None
    filename = path_str.split('/')[-1]
    # 移除扩展名
    frame = filename.rsplit('.', 1)[0]
    # 如果有下划线，取第一部分（时间戳）
    if '_' in frame:
        frame = frame.split('_')[0]
    return frame


def load_csv_with_per_frame_miou(csv_path, dataset_name='orfd2road'):
    """加载 CSV 并计算每帧的 mIoU"""
    df = pd.read_csv(csv_path)

    # 处理列名
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'path'})

    # 检测 CSV 格式
    conf_cols = [c for c in df.columns if '-' in c and all(part.isdigit() for part in c.split('-'))]
    class_summary_cols = [c for c in df.columns if c.startswith('class') and c.endswith('_intersect')]

    results = []

    if conf_cols:
        # 显式混淆矩阵格式 (off-net, m2f2)
        class_ids = sorted({int(part) for col in conf_cols for part in col.split('-')})
        num_classes = max(class_ids) + 1

        for idx, row in df.iterrows():
            mat = np.zeros((num_classes, num_classes), dtype=np.float64)
            for col in conf_cols:
                i, j = map(int, col.split('-'))
                mat[i, j] = row[col]

            miou = calculate_miou(mat)

            # 获取路径
            path_col = 'path' if 'path' in df.columns else df.columns[0]
            path = row[path_col] if path_col in row else str(idx)
            scene = row['scene'] if 'scene' in row else None

            # 如果没有 scene 列，从路径提取
            if scene is None or pd.isna(scene):
                scene, frame = extract_scene_and_frame(str(path), 'conf')
            else:
                scene = str(scene).replace('_', '-')
                # 使用统一的帧名提取函数
                frame = extract_frame_from_path(str(path)) if isinstance(path, str) else str(idx)

            results.append({
                'path': path,
                'scene': scene,
                'frame': frame,
                'miou': miou
            })

    elif class_summary_cols:
        # Per-class summary 格式 (tqt, rod)
        for idx, row in df.iterrows():
            tp_pos = row['class1_intersect']
            pred_pos = row['class1_pred_label']
            label_pos = row['class1_label']
            tn = row['class0_intersect']

            fp_pos = pred_pos - tp_pos
            fn_pos = label_pos - tp_pos

            mat = np.array([
                [tn, fp_pos],
                [fn_pos, tp_pos],
            ], dtype=np.float64)

            miou = calculate_miou(mat)

            # 获取路径和场景
            path_col = 'path' if 'path' in df.columns else df.columns[0]
            path = row[path_col] if path_col in row else str(idx)
            scene = row['scene'] if 'scene' in row else None

            # 检查 scene 是否是完整路径（rod 的情况）
            scene_str = str(scene) if scene is not None and not pd.isna(scene) else ''
            if scene is None or pd.isna(scene) or 'testing/' in scene_str or '/' in scene_str:
                # 从路径或 scene 列中提取场景名
                extract_from = scene_str if 'testing/' in scene_str else str(path)
                scene, frame = extract_scene_and_frame(extract_from, 'summary')
                # 如果从 scene 列提取失败，尝试从 path 提取
                if scene is None:
                    scene, frame = extract_scene_and_frame(str(path), 'summary')
            else:
                # 处理场景名格式
                if scene_str.startswith(('x', 'y')) and len(scene_str) > 1:
                    scene = scene_str[1:].replace('_', '-')
                else:
                    scene = scene_str.replace('_', '-')
                # 使用统一的帧名提取函数
                frame = extract_frame_from_path(str(path)) if isinstance(path, str) else str(idx)

            results.append({
                'path': path,
                'scene': scene,
                'frame': frame,
                'miou': miou
            })

    return pd.DataFrame(results)


def main():
    # CSV 路径配置
    CSV_PATHS = {
        'off-net': "/root/tqdm/offnet_test_best_orfd2road.csv",
        'm2f2': "/root/tqdm/m2f2net-result-orfd2road.csv",
        'rod': "/root/ROD/orfd2road_eval_results.csv",
        'tqt': "/root/tqdm/tqttesting_eval_file_stats_20251223_185714.csv",
    }

    dataset_name = 'orfd2road'
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])
    print(f"未知场景列表: {unknown_scenes}")

    # 加载所有 CSV 并计算每帧 mIoU
    all_data = {}
    for method, csv_path in CSV_PATHS.items():
        print(f"\n加载 {method}: {csv_path}")
        df = load_csv_with_per_frame_miou(csv_path, dataset_name)
        all_data[method] = df
        print(f"  共 {len(df)} 帧, 场景: {df['scene'].nunique()}")
        print(f"  场景列表: {sorted(df['scene'].dropna().unique())[:10]}...")

    # 获取 TQT 数据
    tqt_df = all_data['tqt']
    baseline_methods = ['off-net', 'm2f2', 'rod']

    # 筛选未知场景
    def is_unknown(scene):
        if scene is None or pd.isna(scene):
            return False
        for us in unknown_scenes:
            if us in str(scene):
                return True
        return False

    tqt_unknown = tqt_df[tqt_df['scene'].apply(is_unknown)].copy()
    print(f"\n未知场景帧数 (TQT): {len(tqt_unknown)}")

    # 对于每一帧，计算与 baseline 最大值的差值
    results = []

    for idx, tqt_row in tqt_unknown.iterrows():
        tqt_miou = tqt_row['miou']
        tqt_scene = tqt_row['scene']
        tqt_frame = tqt_row['frame']
        tqt_path = tqt_row['path']

        # 查找对应的 baseline 结果
        baseline_mious = {}
        for method in baseline_methods:
            method_df = all_data[method]
            # 按场景和帧匹配
            matched = method_df[
                (method_df['scene'].apply(lambda x: tqt_scene in str(x) if x else False)) &
                (method_df['frame'] == tqt_frame)
            ]
            if len(matched) > 0:
                baseline_mious[method] = matched.iloc[0]['miou']
            else:
                # 尝试更宽松的匹配
                matched = method_df[method_df['frame'] == tqt_frame]
                if len(matched) > 0:
                    baseline_mious[method] = matched.iloc[0]['miou']

        if baseline_mious:
            max_baseline = max(baseline_mious.values())
            max_baseline_method = max(baseline_mious, key=baseline_mious.get)
            diff = (tqt_miou - max_baseline) * 100  # 转为百分比

            results.append({
                'scene': tqt_scene,
                'frame': tqt_frame,
                'path': tqt_path,
                'tqt_miou': tqt_miou * 100,
                'off-net_miou': baseline_mious.get('off-net', None),
                'm2f2_miou': baseline_mious.get('m2f2', None),
                'rod_miou': baseline_mious.get('rod', None),
                'max_baseline': max_baseline * 100,
                'max_baseline_method': max_baseline_method,
                'diff': diff
            })

    # 按差值从大到小排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('diff', ascending=False)

    print("\n" + "=" * 150)
    print("未知场景每帧 mIoU 差值 (TQT - max(off-net, m2f2, rod))，按差值从大到小排序")
    print("=" * 150)

    print(f"\n{'Scene':<20} {'Frame':<20} {'TQT':>10} {'off-net':>10} {'m2f2':>10} {'rod':>10} {'MaxBL':>10} {'From':>8} {'Diff':>12}")
    print("-" * 150)

    for _, row in results_df.iterrows():
        offnet = f"{row['off-net_miou']*100:.2f}%" if row['off-net_miou'] else "-"
        m2f2 = f"{row['m2f2_miou']*100:.2f}%" if row['m2f2_miou'] else "-"
        rod = f"{row['rod_miou']*100:.2f}%" if row['rod_miou'] else "-"

        diff_str = f"{row['diff']:+.2f}%"

        print(f"{row['scene']:<20} {row['frame']:<20} {row['tqt_miou']:>9.2f}% {offnet:>10} {m2f2:>10} {rod:>10} {row['max_baseline']:>9.2f}% {row['max_baseline_method']:>8} {diff_str:>12}")

    print("-" * 150)

    # 统计
    positive = (results_df['diff'] > 0).sum()
    negative = (results_df['diff'] < 0).sum()
    zero = (results_df['diff'] == 0).sum()
    avg_diff = results_df['diff'].mean()

    print(f"\n统计: 提升 {positive} 帧, 下降 {negative} 帧, 持平 {zero} 帧")
    print(f"平均差异: {avg_diff:+.2f}%")
    print(f"最大提升: {results_df['diff'].max():+.2f}%")
    print(f"最大下降: {results_df['diff'].min():+.2f}%")

    # 按场景分组统计
    print("\n" + "=" * 100)
    print("按场景分组统计")
    print("=" * 100)

    for scene in sorted(results_df['scene'].unique()):
        scene_df = results_df[results_df['scene'] == scene]
        scene_avg = scene_df['diff'].mean()
        scene_pos = (scene_df['diff'] > 0).sum()
        scene_neg = (scene_df['diff'] < 0).sum()
        print(f"{scene:<25} 帧数: {len(scene_df):>4}, 平均差: {scene_avg:+.2f}%, 提升: {scene_pos}, 下降: {scene_neg}")

    # 保存为 CSV 文件
    output_csv = "/root/tqdm/compare_frame_miou_results.csv"
    results_df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")


if __name__ == '__main__':
    main()
