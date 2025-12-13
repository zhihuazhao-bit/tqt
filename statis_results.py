from pathlib import Path

import numpy as np
import pandas as pd

from util.util import getScores_self

# 支持传入多个 CSV，方便批量评估
CSV_PATHS = [
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251109_231619.csv",
    # "/root/ORFD/testresults/ROAD-all/test_best/confusion_matrices_test_best.csv",
    # "/root/M2F2-Net/ckpts/road3d-m2f2net-all/results-plus/result.csv",

    "/root/ORFD/testresults/ORFD-all/test_best/confusion_matrices_test_best.csv",
    "/root/M2F2-Net/ckpts/orfd-m2f2net-all/results-plus/result.csv",

    # # pixel-add
    # "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-r2r-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-add/testing_eval_file_stats_20251109_205509.csv",
    # # pixel-proj
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251109_221500.csv",
    # # context-add
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251109_224609.csv",
    # # context-proj
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251109_231619.csv",
    # # pixel-proj-nocross-attn
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251111_075927.csv",
    # # pixel-proj-nopropmt-cls
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251111_082911.csv",
    # # pixel-proj-useeva
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251111_085911.csv",
    # # pixel-concat
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251111_092812.csv"

    # pixel-proj-nocross-attn
    # "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-r2r-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nocrossattn/testing_eval_file_stats_20251112_100611.csv",
    # pixel-proj-nopropmt-cls
    # "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-r2r-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-nopromptcls/testing_eval_file_stats_20251112_103543.csv",
    # pixel-proj-useeva
    # "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-r2r-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj-useeva/testing_eval_file_stats_20251112_110502.csv",
    # 
    "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-o2o-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj/testing_eval_file_stats_20251117_102620.csv"
    # "/root/tqdm/work_dirs/test/tqt_eva_vit-b_1e-5_5k-r2r-512-all-traversable-pixel-proj-cls-prefix-224x224-pixel-proj/testing_eval_file_stats_20251117_105215.csv"
    # orfd-pixel-proj
    # "/root/tqdm/csv_result/testing_eval_file_stats_20251112_111221.csv",
]

csv_path_objs = [Path(p) for p in CSV_PATHS]
missing_files = [str(path) for path in csv_path_objs if not path.exists()]
if missing_files:
    raise FileNotFoundError("找不到以下 CSV 文件:\n" + "\n".join(missing_files))


def analyze_csv(csv_path: Path):
    print("\n" + "-" * 80)
    print(f"Processing: {csv_path}")
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
        df['scene'] = df['scene'].apply(lambda x: x[1:].replace("_", "-") if isinstance(x, str) else x)

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

    DATASET_UNKNOWN_SCENES = {
        # 'road3d': ['2021-0403-1744', '0602-1107', '2021-0222-1743', '2021-0403-1858'],
        'orfd': ['0609-1923', '2021-0223-1756'],
    }

    csv_lower = str(csv_path).lower()
    candidate_datasets = [name for name in DATASET_UNKNOWN_SCENES if name in csv_lower]

    if not candidate_datasets:
        scene_set = set(df['scene'])
        for name, scenes in DATASET_UNKNOWN_SCENES.items():
            if scene_set.intersection(scenes):
                candidate_datasets.append(name)

    if not candidate_datasets:
        candidate_datasets = list(DATASET_UNKNOWN_SCENES.keys())

    overall_mat = subset_matrix(df)

    def print_metrics(tag: str, mat: np.ndarray):
        mAcc, mrecall, mf1, miou, fwIoU, prec_road, rec_road, f1_road, iou_road = getScores_self(mat)
        print(f"[{tag}] mAcc={mAcc:.3f}, mRecall={mrecall:.3f}, mF1={mf1:.3f}, mIoU={miou:.3f}, fwIoU={fwIoU:.3f}, "
              f"prec_road={prec_road:.3f}, rec_road={rec_road:.3f}, f1_road={f1_road:.3f}, iou_road={iou_road:.3f}")

    print(f"Loaded {len(df)} image rows over {df['scene'].nunique()} scenes")
    print_metrics('overall', overall_mat)

    for dataset_name in candidate_datasets:
        unknown_scenes = DATASET_UNKNOWN_SCENES[dataset_name]
        unknown_mask = df['scene'].isin(unknown_scenes)
        present_unknown = sorted(set(df.loc[unknown_mask, 'scene']))
        missing_unknown = [scene for scene in unknown_scenes if scene not in present_unknown]

        if not present_unknown:
            print(f"[{dataset_name}] 未找到指定的 unknown 场景，可能该 CSV 不属于此数据集，跳过。")
            continue

        known_mat = subset_matrix(df[~unknown_mask])
        unknown_mat = subset_matrix(df[unknown_mask])
        dataset_mat = subset_matrix(df)

        print(f"\n[{dataset_name}] 指标统计 (scenes: {df['scene'].nunique()}, unknown: {len(present_unknown)})")
        if missing_unknown:
            print("  [Warn] 下列 unknown 场景缺失:", ', '.join(missing_unknown))
        print_metrics(f'{dataset_name}-overall', dataset_mat)
        print_metrics(f'{dataset_name}-known', known_mat)
        print_metrics(f'{dataset_name}-unknown', unknown_mat)


for csv_path in csv_path_objs:
    analyze_csv(csv_path)