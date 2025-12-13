#!/usr/bin/env python
"""
验证 TQT 纯净版本 (use_sne=False, prompt_cls=False) 与 TQDM 输出一致性

验证方案:
- TQDM: tqdm_eva_vit-b_1e-5_5k-o2o-224.py (ORFD 224, 前 10 iter)
- TQT:  tqt_eva_vit-b_1e-5_5k-o2o-224-baseline.py (use_sne=False, 相同数据集/权重)

预期: 两者的 loss 和梯度应完全一致
"""

import subprocess
import sys
import os
import argparse

# 配置文件列表 - 用于验证 TQT baseline 与 TQDM 一致性
CONFIGS = [
    # (config_path, description)
    ('configs/tqdm/tqdm_eva_vit-b_1e-5_5k-o2o-224.py', 'TQDM ORFD 224 (reference)'),
    ('configs/tqt/tqt_eva_vit-b_1e-5_5k-o2o-224-baseline.py', 'TQT ORFD 224 baseline (use_sne=False)'),
]

def run_training(config_path, gpu_id=0, max_iters=10):
    """运行训练"""
    cmd = [
        'python', 'train.py',
        '--config', config_path,
        f'--gpu-id={gpu_id}',
        '--seed=2023',  # 固定随机种子确保可复现
        '--deterministic',
    ]
    
    print(f"\n{'='*60}")
    print(f"Running: {config_path}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description='验证 TQT 与 TQDM 一致性')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--config', type=int, default=-1, 
                        help='只运行指定索引的配置 (0-4), -1 表示全部运行')
    parser.add_argument('--list', action='store_true', help='列出所有配置')
    args = parser.parse_args()
    
    if args.list:
        print("\n可用配置列表:")
        print("-" * 60)
        for i, (cfg, desc) in enumerate(CONFIGS):
            print(f"  [{i}] {desc}")
            print(f"      {cfg}")
        print("-" * 60)
        print("\n使用示例:")
        print("  python verify_tqt_tqdm.py --gpu 0           # 运行所有配置")
        print("  python verify_tqt_tqdm.py --gpu 0 --config 0  # 只运行第一个配置")
        return
    
    # 确定要运行的配置
    if args.config >= 0:
        if args.config >= len(CONFIGS):
            print(f"错误: 配置索引 {args.config} 超出范围 (0-{len(CONFIGS)-1})")
            return
        configs_to_run = [CONFIGS[args.config]]
    else:
        configs_to_run = CONFIGS
    
    print(f"\n将运行 {len(configs_to_run)} 个配置:")
    for cfg, desc in configs_to_run:
        print(f"  - {desc}: {cfg}")
    
    # 依次运行
    results = []
    for cfg, desc in configs_to_run:
        if not os.path.exists(cfg):
            print(f"\n警告: 配置文件不存在: {cfg}")
            results.append((desc, 'NOT_FOUND'))
            continue
            
        ret = run_training(cfg, args.gpu)
        status = 'SUCCESS' if ret == 0 else f'FAILED (code={ret})'
        results.append((desc, status))
    
    # 打印结果汇总
    print(f"\n{'='*60}")
    print("运行结果汇总:")
    print('='*60)
    for desc, status in results:
        print(f"  {desc}: {status}")
    print('='*60)

if __name__ == '__main__':
    main()
