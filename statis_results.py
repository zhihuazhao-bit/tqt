from pathlib import Path

import numpy as np
import pandas as pd

from statis_ablation_results import getScores_self
from utils.scene_config import DATASET_UNKNOWN_SCENES, ABNORMAL_SCENES

# 按数据集分类的 CSV 路径
CSV_PATHS = {
    'road3d': [
        "/root/ORFD/testresults/ROAD-all/test_best/confusion_matrices_test_best.csv",
        "/root/M2F2-Net/ckpts/road3d-m2f2net-all/results-plus/result.csv",
        "/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/20251218_1320/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_road_learnableT_promptTau/testing_eval_file_stats_20251219_101507.csv",
        "/root/ROD/road2road_eval_results.csv",  # ROD model (Road2Road)
    ],
    'orfd': [
        "/root/ORFD/testresults/ORFD-all/test_best2orfd/confusion_matrices_test_best.csv",
        "/root/M2F2-Net/ckpts/orfd-m2f2net-all/results-plus/result.csv",
        "/root/tqdm/work_dirs/ablation_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/20251219_1354/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/test_results/exp_512_eva02_sneotTrue_patchfpn_pisup_promptSoft_no_cos_mean_prob_softunion_learnableT_promptTau/testing_eval_file_stats_20251220_085925.csv",
        "/root/ROD/orfd2orfd_eval_results.csv",  # ROD model (ORFD2ORFD)
    ],
    'orfd2road': [
        "/root/tqdm/offnet_test_best_orfd2road.csv",
        "/root/tqdm/m2f2net-result-orfd2road.csv",
        "/root/tqdm/tqttesting_eval_file_stats_20251223_185714.csv",
        "/root/ROD/orfd2road_eval_results.csv",  # ROD model (ORFD2Road cross-domain)
    ]
}

# 检查文件是否存在
for dataset_name, paths in CSV_PATHS.items():
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"[{dataset_name}] 找不到 CSV 文件: {p}")


def analyze_csv(csv_path: Path, dataset_name: str):
    print("\n" + "-" * 80)
    print(f"[{dataset_name.upper()}] Processing: {csv_path}")
    print("-" * 80)

    df = pd.read_csv(csv_path)
    if 'Unnamed: 0' in df.columns:
        df = df.rename(columns={'Unnamed: 0': 'path'})

    conf_cols = [c for c in df.columns if '-' in c and all(part.isdigit() for part in c.split('-'))]
    class_summary_cols = [c for c in df.columns if c.startswith('class') and c.endswith('_intersect')]

    if conf_cols:
        format_type = 'explicit_conf'
    elif class_summary_cols:
        format_type = 'per_class_summary'
    else:
        raise ValueError("无法识别 CSV 格式：既没有 'i-j' 混淆矩阵列，也没有 'classX_intersect' 列")

    if format_type == 'explicit_conf':
        class_ids = sorted({int(part) for col in conf_cols for part in col.split('-')})
        num_classes = max(class_ids) + 1

        def subset_matrix(sub_df: pd.DataFrame) -> np.ndarray:
            if sub_df.empty:
                return np.zeros((num_classes, num_classes), dtype=np.float64)
            mat = np.zeros((num_classes, num_classes), dtype=np.float64)
            series = sub_df[conf_cols].sum()
            for key, value in series.items():
                i, j = map(int, key.split('-'))
                mat[i, j] = value
            return mat

    elif format_type == 'per_class_summary':
        class_ids = sorted(int(col[len('class'):].split('_')[0]) for col in class_summary_cols)
        num_classes = len(class_ids)
        if num_classes != 2:
            raise NotImplementedError("per-class summary 格式当前仅支持二分类")

        required_suffixes = ['_intersect', '_pred_label', '_label']
        for cid in class_ids:
            for suffix in required_suffixes:
                col = f'class{cid}{suffix}'
                if col not in df.columns:
                    raise ValueError(f"缺少列 {col}，无法还原混淆矩阵")

        def extract_scene_name(x):
            """提取场景名，支持完整路径格式（如 ROD）和直接场景名格式（如 TQT）"""
            if not isinstance(x, str):
                return x
            # 如果是完整路径如 "testing/x0613_1627/gt_image/..."，提取场景文件夹名
            if '/' in x:
                x = x.split('/')[1]  # 取第二个部分，即场景文件夹名
            # 移除开头的单字母前缀（如 'x', 'y' 等）并将 "_" 替换为 "-"
            # 场景名格式如: x2021_0223_1756, y0609_1923 等
            if len(x) > 1 and x[0].isalpha() and (x[1].isdigit() or x[1] == '_'):
                return x[1:].replace("_", "-")
            return x.replace("_", "-")

        df['scene'] = df['scene'].apply(extract_scene_name)

        def subset_matrix(sub_df: pd.DataFrame) -> np.ndarray:
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
            if (mat < -1e-6).any():
                raise ValueError("推导出的混淆矩阵存在负值，请检查输入统计是否正确")
            return mat

    else:
        raise AssertionError("未覆盖的 CSV 格式分支")

    # DATASET_UNKNOWN_SCENES 已移至 utils/scene_config.py
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])
    print(f"Unknown 场景列表: {unknown_scenes}")    
    overall_mat = subset_matrix(df)

    def print_metrics(tag: str, mat: np.ndarray):
        mAcc, mrecall, mf1, miou, fwIoU, prec_road, rec_road, f1_road, iou_road = getScores_self(mat)
        print(f"[{tag}] mAcc={mAcc:.4f}, mRecall={mrecall:.4f}, mF1={mf1:.4f}, mIoU={miou:.4f}, fwIoU={fwIoU:.4f}, "
              f"prec_road={prec_road:.4f}, rec_road={rec_road:.4f}, f1_road={f1_road:.4f}, iou_road={iou_road:.4f}")

    print(f"Loaded {len(df)} image rows over {df['scene'].nunique()} scenes")
    print_metrics('overall', overall_mat)

    # 计算 known/unknown 指标
    unknown_mask = df['scene'].isin(unknown_scenes)
    present_unknown = sorted(set(df.loc[unknown_mask, 'scene']))
    missing_unknown = [scene for scene in unknown_scenes if scene not in present_unknown]

    known_mat = subset_matrix(df[~unknown_mask])
    unknown_mat = subset_matrix(df[unknown_mask])

    print(f"\n指标统计 (scenes: {df['scene'].nunique()}, unknown: {len(present_unknown)})")
    if missing_unknown:
        print("  [Warn] 下列 unknown 场景缺失:", ', '.join(missing_unknown))
    print_metrics('known', known_mat)
    print_metrics('unknown', unknown_mat)

    # 每个 unknown 场景的单独指标
    print(f"\n  [Per-Scene mIoU] Unknown 场景详细:")
    for scene in unknown_scenes:
        scene_mask = df['scene'] == scene
        if scene_mask.sum() > 0:
            scene_mat = subset_matrix(df[scene_mask])
            _, _, _, scene_miou, _, _, _, _, _ = getScores_self(scene_mat)
            num_samples = scene_mask.sum()
            print(f"    {scene:20s} -> mIoU: {scene_miou*100:.4f}%  (n={num_samples})")
        else:
            print(f"    {scene:20s} -> (not found)")

    # 返回所有场景的 mIoU 用于汇总
    all_scenes = sorted(df['scene'].unique())
    scene_miou_dict = {}
    scene_frame_data = {}  # 每帧的详细数据
    for scene in all_scenes:
        scene_mask = df['scene'] == scene
        if scene_mask.sum() > 0:
            scene_mat = subset_matrix(df[scene_mask])
            _, _, _, scene_miou, _, _, _, _, _ = getScores_self(scene_mat)
            scene_miou_dict[scene] = scene_miou

            # 计算每帧的 mIoU
            frame_mious = []
            scene_df = df[scene_mask]
            for idx, row in scene_df.iterrows():
                frame_mat = subset_matrix(scene_df.loc[[idx]])
                _, _, _, frame_miou, _, _, _, _, _ = getScores_self(frame_mat)
                # 获取帧路径/名称
                if 'path' in row:
                    frame_name = Path(row['path']).stem if isinstance(row['path'], str) else str(idx)
                else:
                    frame_name = str(idx)
                frame_mious.append({'name': frame_name, 'miou': frame_miou})
            scene_frame_data[scene] = frame_mious

    return scene_miou_dict, scene_frame_data


def print_scene_comparison(dataset_name: str, results: dict, unknown_scenes: list):
    """打印所有场景的 mIoU 对比表格"""
    if not results:
        return

    # 获取所有场景
    all_scenes = set()
    for scene_dict in results.values():
        all_scenes.update(scene_dict.keys())
    all_scenes = sorted(all_scenes)

    # 获取方法名（从路径提取简短名称）
    method_names = []
    for path in results.keys():
        if 'ORFD/testresults' in path or 'ROAD-all' in path or 'offnet_test_best' in path:
            method_names.append('ORFD-Baseline')
        elif 'M2F2-Net' in path or 'm2f2net-result' in path:
            method_names.append('M2F2-Net')
        elif '/ROD/' in path:
            method_names.append('ROD')
        elif 'work_dirs' in path or 'tqttesting_eval' in path or 'testing_eval_file_stats' in path:
            method_names.append('Ours')
        else:
            method_names.append(Path(path).stem[:20])

    print("\n" + "=" * 120)
    print(f"[{dataset_name.upper()}] 所有场景 mIoU 对比")
    print("=" * 120)

    # 表头
    header = f"{'Scene':<25} {'Type':<8}"
    for name in method_names:
        header += f" {name:>15}"
    print(header)
    print("-" * 120)

    # 每个场景的数据
    for scene in all_scenes:
        is_unknown = scene in unknown_scenes
        scene_type = "Unknown" if is_unknown else "Known"
        row = f"{scene:<25} {scene_type:<8}"
        for path in results.keys():
            miou = results[path].get(scene, None)
            if miou is not None:
                row += f" {miou*100:>14.4f}%"
            else:
                row += f" {'-':>15}"
        print(row)

    # 汇总行
    print("-" * 120)

    # Known 平均
    row = f"{'[Known Avg]':<25} {'':<8}"
    for path in results.keys():
        known_mious = [results[path].get(s, None) for s in all_scenes if s not in unknown_scenes]
        known_mious = [m for m in known_mious if m is not None]
        if known_mious:
            row += f" {np.mean(known_mious)*100:>14.4f}%"
        else:
            row += f" {'-':>15}"
    print(row)

    # Unknown 平均
    row = f"{'[Unknown Avg]':<25} {'':<8}"
    for path in results.keys():
        unknown_mious = [results[path].get(s, None) for s in all_scenes if s in unknown_scenes]
        unknown_mious = [m for m in unknown_mious if m is not None]
        if unknown_mious:
            row += f" {np.mean(unknown_mious)*100:>14.4f}%"
        else:
            row += f" {'-':>15}"
    print(row)

    # Overall 平均
    row = f"{'[Overall Avg]':<25} {'':<8}"
    for path in results.keys():
        all_mious = [results[path].get(s, None) for s in all_scenes]
        all_mious = [m for m in all_mious if m is not None]
        if all_mious:
            row += f" {np.mean(all_mious)*100:>14.4f}%"
        else:
            row += f" {'-':>15}"
    print(row)

    # 计算 Ours 与 Best Baseline 的差异
    print("\n" + "=" * 150)
    print(f"[{dataset_name.upper()}] Ours vs Best Baseline 差异 (按差异从小到大排序)")
    print("=" * 150)

    # 找到 Ours 和 Baseline 的路径
    ours_path = None
    baseline_paths = []
    for path in results.keys():
        if 'work_dirs' in path or 'tqttesting_eval' in path or 'testing_eval_file_stats' in path:
            ours_path = path
        else:
            baseline_paths.append(path)

    if ours_path is None or not baseline_paths:
        print("  无法找到 Ours 或 Baseline 结果")
        return

    # 计算每个场景的差异
    diff_list = []
    for scene in all_scenes:
        ours_miou = results[ours_path].get(scene, None)
        if ours_miou is None:
            continue

        # 收集所有方法的 mIoU
        all_methods_miou = {}
        for path in results.keys():
            miou = results[path].get(scene, None)
            if 'ORFD/testresults' in path or 'ROAD-all' in path or 'offnet_test_best' in path:
                all_methods_miou['ORFD-BL'] = miou
            elif 'M2F2-Net' in path or 'm2f2net-result' in path:
                all_methods_miou['M2F2'] = miou
            elif '/ROD/' in path:
                all_methods_miou['ROD'] = miou
            elif 'work_dirs' in path or 'tqttesting_eval' in path or 'testing_eval_file_stats' in path:
                all_methods_miou['Ours'] = miou

        # 找到 baseline 中的最佳值
        best_baseline = None
        best_baseline_name = None
        for bp in baseline_paths:
            baseline_miou = results[bp].get(scene, None)
            if baseline_miou is not None:
                if best_baseline is None or baseline_miou > best_baseline:
                    best_baseline = baseline_miou
                    if 'ORFD/testresults' in bp or 'ROAD-all' in bp or 'offnet_test_best' in bp:
                        best_baseline_name = 'ORFD-BL'
                    elif 'M2F2-Net' in bp or 'm2f2net-result' in bp:
                        best_baseline_name = 'M2F2'
                    elif '/ROD/' in bp:
                        best_baseline_name = 'ROD'
                    else:
                        best_baseline_name = Path(bp).stem[:10]

        if best_baseline is not None:
            diff = (ours_miou - best_baseline) * 100  # 转为百分比差异
            is_unknown = scene in unknown_scenes
            diff_list.append({
                'scene': scene,
                'type': 'Unk' if is_unknown else 'Kn',
                'ours': ours_miou * 100,
                'orfd_bl': all_methods_miou.get('ORFD-BL', None),
                'm2f2': all_methods_miou.get('M2F2', None),
                'rod': all_methods_miou.get('ROD', None),
                'best_bl': best_baseline * 100,
                'best_bl_name': best_baseline_name,
                'diff': diff
            })

    # 按差异从小到大排序
    diff_list.sort(key=lambda x: x['diff'])

    # 打印表头
    print(f"{'Scene':<25} {'Type':<5} {'ORFD-BL':>10} {'M2F2':>10} {'ROD':>10} {'Ours':>10} {'BestBL':>10} {'From':>8} {'Diff':>12}")
    print("-" * 150)

    for item in diff_list:
        diff_str = f"{item['diff']:+.4f}%"
        if item['diff'] < 0:
            diff_str = f"\033[91m{diff_str}\033[0m"  # 红色表示负差异
        elif item['diff'] > 2:
            diff_str = f"\033[92m{diff_str}\033[0m"  # 绿色表示大于2%的提升

        orfd_str = f"{item['orfd_bl']*100:.4f}%" if item['orfd_bl'] else "-"
        m2f2_str = f"{item['m2f2']*100:.4f}%" if item['m2f2'] else "-"
        rod_str = f"{item['rod']*100:.4f}%" if item['rod'] else "-"

        print(f"{item['scene']:<25} {item['type']:<5} {orfd_str:>12} {m2f2_str:>12} {rod_str:>12} {item['ours']:>11.4f}% {item['best_bl']:>11.4f}% {item['best_bl_name']:>8} {diff_str:>14}")

    print("-" * 150)

    # 统计正负差异数量
    positive = sum(1 for x in diff_list if x['diff'] > 0)
    negative = sum(1 for x in diff_list if x['diff'] < 0)
    zero = sum(1 for x in diff_list if x['diff'] == 0)
    avg_diff = np.mean([x['diff'] for x in diff_list]) if diff_list else 0

    print(f"统计: 提升 {positive} 个场景, 下降 {negative} 个场景, 持平 {zero} 个场景")
    print(f"平均差异: {avg_diff:+.4f}%")

    # 返回低于 85% 的场景列表
    low_miou_scenes = [item['scene'] for item in diff_list if item['ours'] < 85]
    return low_miou_scenes


# 数据集 unknown 场景配置
# DATASET_UNKNOWN_SCENES 已移至 utils/scene_config.py

# 按数据集分组处理
for dataset_name, csv_paths in CSV_PATHS.items():
    print("\n" + "=" * 80)
    print(f"Dataset: {dataset_name.upper()}")
    print("=" * 80)

    # 收集每个 CSV 的场景 mIoU 和每帧数据
    all_results = {}
    all_frame_data = {}
    for csv_path in csv_paths:
        scene_miou, frame_data = analyze_csv(Path(csv_path), dataset_name)
        all_results[csv_path] = scene_miou
        all_frame_data[csv_path] = frame_data

    # 打印汇总对比表格
    unknown_scenes = DATASET_UNKNOWN_SCENES.get(dataset_name, [])
    low_miou_scenes = print_scene_comparison(dataset_name, all_results, unknown_scenes)

    # 对低于 85% mIoU 的场景，展示每帧详细结果
    if low_miou_scenes:
        print("\n" + "=" * 100)
        print(f"[{dataset_name.upper()}] 低 mIoU (<85%) 场景每帧详细分析")
        print("=" * 100)

        # 找到 Ours 的路径
        ours_path = None
        for path in all_results.keys():
            if 'tqdm' in path or 'work_dirs' in path:
                ours_path = path
                break

        if ours_path and ours_path in all_frame_data:
            for scene in low_miou_scenes:
                if scene in all_frame_data[ours_path]:
                    frame_list = all_frame_data[ours_path][scene]
                    # 按 mIoU 从小到大排序
                    frame_list_sorted = sorted(frame_list, key=lambda x: x['miou'])

                    scene_miou = all_results[ours_path].get(scene, 0) * 100
                    print(f"\n场景: {scene} (整体 mIoU: {scene_miou:.4f}%, 共 {len(frame_list)} 帧)")
                    print("-" * 100)
                    print(f"{'Rank':<6} {'Frame':<60} {'mIoU':>10}")
                    print("-" * 100)

                    # 显示最差的 20 帧
                    for i, frame in enumerate(frame_list_sorted[:20]):
                        miou_str = f"{frame['miou']*100:.4f}%"
                        if frame['miou'] < 0.7:
                            miou_str = f"\033[91m{miou_str}\033[0m"  # 红色
                        elif frame['miou'] < 0.8:
                            miou_str = f"\033[93m{miou_str}\033[0m"  # 黄色
                        print(f"{i+1:<6} {frame['name']:<60} {miou_str:>10}")

                    if len(frame_list_sorted) > 20:
                        print(f"... 省略 {len(frame_list_sorted) - 20} 帧")

                    # 统计分布
                    below_70 = sum(1 for f in frame_list if f['miou'] < 0.7)
                    below_80 = sum(1 for f in frame_list if 0.7 <= f['miou'] < 0.8)
                    below_85 = sum(1 for f in frame_list if 0.8 <= f['miou'] < 0.85)
                    above_85 = sum(1 for f in frame_list if f['miou'] >= 0.85)
                    print(f"\n分布: <70%: {below_70}, 70-80%: {below_80}, 80-85%: {below_85}, >=85%: {above_85}")