#!/usr/bin/env python3
"""
Compare mIoU metrics between two test result CSV files on a per-scene basis.

Usage:
    python compare_miou.py <file1> <file2>
    python compare_miou.py confusion_matrices_test_best.csv testing_eval_file_stats_20251223_185714.csv

    # With unknown scenes (suffixes)
    python compare_miou.py <file1> <file2> --unknown 1309 1924 1923 1736 1857

Supported formats:
    - Confusion matrix format: columns [0-0, 0-1, 1-0, 1-1, scene]
    - Intersection/Union format: columns [class0_intersect, class0_union, class1_intersect, class1_union, scene]
"""

import argparse
import pandas as pd
import numpy as np
import sys

# Default unknown scene suffixes
DEFAULT_UNKNOWN_SUFFIXES = [
    # '1309', 
    '1924', '1923', 
    '1736', '1857'
    ]


def normalize_scene(s):
    """Normalize scene names by removing prefixes and replacing underscores."""
    s = str(s)
    if s.startswith('x') or s.startswith('y'):
        s = s[1:]
    return s.replace('_', '-')


def is_unknown_scene(scene, unknown_suffixes):
    """Check if a scene is unknown based on suffixes."""
    for suffix in unknown_suffixes:
        if scene.endswith(suffix):
            return True
    return False


def calc_miou_from_cm(df):
    """Calculate mIoU from confusion matrix format (aggregated)."""
    total = df[['0-0', '0-1', '1-0', '1-1']].sum()
    iou0 = total['0-0'] / (total['0-0'] + total['0-1'] + total['1-0'])
    iou1 = total['1-1'] / (total['1-1'] + total['0-1'] + total['1-0'])
    return (iou0 + iou1) / 2


def calc_miou_from_iu(df):
    """Calculate mIoU from intersection/union format (aggregated)."""
    total = df[['class0_intersect', 'class0_union', 'class1_intersect', 'class1_union']].sum()
    iou0 = total['class0_intersect'] / total['class0_union']
    iou1 = total['class1_intersect'] / total['class1_union']
    return (iou0 + iou1) / 2


def detect_format(df):
    """Detect the CSV format based on columns."""
    cols = df.columns.tolist()
    if '0-0' in cols and '1-1' in cols:
        return 'confusion_matrix'
    elif 'class0_intersect' in cols and 'class0_union' in cols:
        return 'intersection_union'
    else:
        raise ValueError(f"Unknown format. Columns: {cols}")


def calculate_miou_confusion_matrix(df):
    """Calculate per-scene mIoU from confusion matrix format."""
    df['scene_norm'] = df['scene'].apply(normalize_scene)

    scene_stats = df.groupby('scene_norm').agg({
        '0-0': 'sum',
        '0-1': 'sum',
        '1-0': 'sum',
        '1-1': 'sum',
        'scene': 'first'
    }).reset_index()

    # IoU_class0 = TN / (TN + FP + FN)
    # IoU_class1 = TP / (TP + FP + FN)
    scene_stats['IoU_class0'] = scene_stats['0-0'] / (scene_stats['0-0'] + scene_stats['0-1'] + scene_stats['1-0'])
    scene_stats['IoU_class1'] = scene_stats['1-1'] / (scene_stats['1-1'] + scene_stats['0-1'] + scene_stats['1-0'])
    scene_stats['mIoU'] = (scene_stats['IoU_class0'] + scene_stats['IoU_class1']) / 2

    # Overall
    overall = df[['0-0', '0-1', '1-0', '1-1']].sum()
    iou0 = overall['0-0'] / (overall['0-0'] + overall['0-1'] + overall['1-0'])
    iou1 = overall['1-1'] / (overall['1-1'] + overall['0-1'] + overall['1-0'])
    overall_miou = (iou0 + iou1) / 2

    return scene_stats[['scene_norm', 'scene', 'mIoU']], overall_miou


def calculate_miou_intersection_union(df):
    """Calculate per-scene mIoU from intersection/union format."""
    df['scene_norm'] = df['scene'].apply(normalize_scene)

    scene_stats = df.groupby('scene_norm').agg({
        'class0_intersect': 'sum',
        'class0_union': 'sum',
        'class1_intersect': 'sum',
        'class1_union': 'sum',
        'scene': 'first'
    }).reset_index()

    scene_stats['IoU_class0'] = scene_stats['class0_intersect'] / scene_stats['class0_union']
    scene_stats['IoU_class1'] = scene_stats['class1_intersect'] / scene_stats['class1_union']
    scene_stats['mIoU'] = (scene_stats['IoU_class0'] + scene_stats['IoU_class1']) / 2

    # Overall
    overall = df[['class0_intersect', 'class0_union', 'class1_intersect', 'class1_union']].sum()
    iou0 = overall['class0_intersect'] / overall['class0_union']
    iou1 = overall['class1_intersect'] / overall['class1_union']
    overall_miou = (iou0 + iou1) / 2

    return scene_stats[['scene_norm', 'scene', 'mIoU']], overall_miou


def compare_miou(file1, file2, output_csv=None, unknown_suffixes=None):
    """Compare mIoU between two files and print results."""
    # Read files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # Detect formats and calculate mIoU
    fmt1 = detect_format(df1)
    fmt2 = detect_format(df2)

    print(f"File1: {file1} (format: {fmt1})")
    print(f"File2: {file2} (format: {fmt2})")
    print(f"File1 rows: {len(df1)}, File2 rows: {len(df2)}")

    # Add normalized scene and unknown flag
    df1['scene_norm'] = df1['scene'].apply(normalize_scene)
    df2['scene_norm'] = df2['scene'].apply(normalize_scene)

    if unknown_suffixes:
        df1['is_unknown'] = df1['scene_norm'].apply(lambda x: is_unknown_scene(x, unknown_suffixes))
        df2['is_unknown'] = df2['scene_norm'].apply(lambda x: is_unknown_scene(x, unknown_suffixes))

    if fmt1 == 'confusion_matrix':
        stats1, overall1 = calculate_miou_confusion_matrix(df1)
    else:
        stats1, overall1 = calculate_miou_intersection_union(df1)

    if fmt2 == 'confusion_matrix':
        stats2, overall2 = calculate_miou_confusion_matrix(df2)
    else:
        stats2, overall2 = calculate_miou_intersection_union(df2)

    # Merge results
    comparison = pd.merge(
        stats1[['scene_norm', 'scene', 'mIoU']].rename(columns={'mIoU': 'mIoU_file1'}),
        stats2[['scene_norm', 'mIoU']].rename(columns={'mIoU': 'mIoU_file2'}),
        on='scene_norm',
        how='outer'
    )
    comparison['diff'] = comparison['mIoU_file2'] - comparison['mIoU_file1']
    comparison['diff_percent'] = comparison['diff'] * 100

    # Add is_unknown flag to comparison
    if unknown_suffixes:
        comparison['is_unknown'] = comparison['scene_norm'].apply(
            lambda x: is_unknown_scene(x, unknown_suffixes)
        )
    comparison = comparison.sort_values('scene_norm')

    # Print results
    if unknown_suffixes:
        print("\n" + "=" * 95)
        print("Per-Scene mIoU Comparison (Known/Unknown)")
        print("=" * 95)
        print(f"{'Scene':<25} {'Type':<10} {'File1 mIoU':>12} {'File2 mIoU':>12} {'Diff':>10} {'Diff%':>10}")
        print("-" * 95)
    else:
        print("\n" + "=" * 85)
        print("Per-Scene mIoU Comparison")
        print("=" * 85)
        print(f"{'Scene':<25} {'File1 mIoU':>12} {'File2 mIoU':>12} {'Diff':>10} {'Diff%':>10}")
        print("-" * 85)

    for _, row in comparison.iterrows():
        scene = row['scene'] if pd.notna(row['scene']) else row['scene_norm']
        miou1 = f"{row['mIoU_file1']:.4f}" if pd.notna(row['mIoU_file1']) else "N/A"
        miou2 = f"{row['mIoU_file2']:.4f}" if pd.notna(row['mIoU_file2']) else "N/A"
        diff = f"{row['diff']:+.4f}" if pd.notna(row['diff']) else "N/A"
        diff_pct = f"{row['diff_percent']:+.2f}%" if pd.notna(row['diff_percent']) else "N/A"
        if unknown_suffixes:
            scene_type = "Unknown" if row.get('is_unknown', False) else "Known"
            print(f"{scene:<25} {scene_type:<10} {miou1:>12} {miou2:>12} {diff:>10} {diff_pct:>10}")
        else:
            print(f"{scene:<25} {miou1:>12} {miou2:>12} {diff:>10} {diff_pct:>10}")

    print("-" * (95 if unknown_suffixes else 85))

    # Calculate and print Known/Unknown/Overall summary if unknown_suffixes provided
    if unknown_suffixes:
        # Calculate aggregated mIoU for known/unknown
        if fmt1 == 'confusion_matrix':
            known1 = calc_miou_from_cm(df1[~df1['is_unknown']])
            unknown1 = calc_miou_from_cm(df1[df1['is_unknown']])
        else:
            known1 = calc_miou_from_iu(df1[~df1['is_unknown']])
            unknown1 = calc_miou_from_iu(df1[df1['is_unknown']])

        if fmt2 == 'confusion_matrix':
            known2 = calc_miou_from_cm(df2[~df2['is_unknown']])
            unknown2 = calc_miou_from_cm(df2[df2['is_unknown']])
        else:
            known2 = calc_miou_from_iu(df2[~df2['is_unknown']])
            unknown2 = calc_miou_from_iu(df2[df2['is_unknown']])

        known_count = (~comparison['is_unknown']).sum()
        unknown_count = comparison['is_unknown'].sum()

        print("\n" + "=" * 75)
        print("Summary by Scene Type")
        print("=" * 75)
        print(f"{'Category':<25} {'File1 mIoU':>15} {'File2 mIoU':>15} {'Diff':>10} {'Diff%':>10}")
        print("-" * 75)
        print(f"{'Known (' + str(known_count) + ' scenes)':<25} {known1:>15.4f} {known2:>15.4f} {known2-known1:>+10.4f} {(known2-known1)*100:>+9.2f}%")
        print(f"{'Unknown (' + str(unknown_count) + ' scenes)':<25} {unknown1:>15.4f} {unknown2:>15.4f} {unknown2-unknown1:>+10.4f} {(unknown2-unknown1)*100:>+9.2f}%")
        print(f"{'Overall':<25} {overall1:>15.4f} {overall2:>15.4f} {overall2-overall1:>+10.4f} {(overall2-overall1)*100:>+9.2f}%")
        print("=" * 75)

        # Print scene lists
        known_scenes = comparison[~comparison['is_unknown']]['scene_norm'].tolist()
        unknown_scenes = comparison[comparison['is_unknown']]['scene_norm'].tolist()
        print(f"\nKnown scenes ({len(known_scenes)}): {', '.join(known_scenes)}")
        print(f"Unknown scenes ({len(unknown_scenes)}): {', '.join(unknown_scenes)}")
    else:
        print(f"{'Overall':<25} {overall1:>12.4f} {overall2:>12.4f} {overall2-overall1:>+10.4f} {(overall2-overall1)*100:>+9.2f}%")
        print("=" * 85)

    # Summary statistics
    valid_diffs = comparison['diff'].dropna()
    print(f"\nStatistics:")
    print(f"  Matched scenes: {len(valid_diffs)}")
    print(f"  File2 better: {(valid_diffs > 0).sum()} scenes")
    print(f"  File1 better: {(valid_diffs < 0).sum()} scenes")
    print(f"  Mean diff: {valid_diffs.mean():+.4f} ({valid_diffs.mean()*100:+.2f}%)")
    print(f"  Max improvement (File2): {valid_diffs.max():+.4f} ({valid_diffs.max()*100:+.2f}%)")
    print(f"  Max regression (File1): {valid_diffs.min():+.4f} ({valid_diffs.min()*100:+.2f}%)")

    # Save to CSV if requested
    if output_csv:
        # Add summary rows
        summary_rows = [{'scene_norm': 'Overall', 'scene': 'Overall',
                         'mIoU_file1': overall1, 'mIoU_file2': overall2,
                         'diff': overall2 - overall1, 'diff_percent': (overall2 - overall1) * 100}]
        if unknown_suffixes:
            summary_rows.insert(0, {'scene_norm': 'Known', 'scene': 'Known',
                                    'mIoU_file1': known1, 'mIoU_file2': known2,
                                    'diff': known2 - known1, 'diff_percent': (known2 - known1) * 100,
                                    'is_unknown': False})
            summary_rows.insert(1, {'scene_norm': 'Unknown', 'scene': 'Unknown',
                                    'mIoU_file1': unknown1, 'mIoU_file2': unknown2,
                                    'diff': unknown2 - unknown1, 'diff_percent': (unknown2 - unknown1) * 100,
                                    'is_unknown': True})
        result = pd.concat([comparison, pd.DataFrame(summary_rows)], ignore_index=True)
        result.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return comparison, overall1, overall2


def main():
    parser = argparse.ArgumentParser(
        description='Compare mIoU metrics between two test result CSV files.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('file1', help='First CSV file (baseline)')
    parser.add_argument('file2', help='Second CSV file (comparison)')
    parser.add_argument('-o', '--output', help='Output CSV file for comparison results')
    parser.add_argument('-u', '--unknown', nargs='+', metavar='SUFFIX',
                        help='Scene suffixes to mark as unknown (e.g., 1309 1924 1923 1736 1857)')
    parser.add_argument('--default-unknown', action='store_true',
                        help='Use default unknown suffixes: 1309 1924 1923 1736 1857')

    args = parser.parse_args()

    # Determine unknown suffixes
    unknown_suffixes = None
    if args.default_unknown:
        unknown_suffixes = DEFAULT_UNKNOWN_SUFFIXES
    elif args.unknown:
        unknown_suffixes = args.unknown

    try:
        compare_miou(args.file1, args.file2, args.output, unknown_suffixes)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
