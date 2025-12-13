#!/usr/bin/env python3
"""
TQT 消融实验批量运行脚本

实验设计:
  A1: 224 + EVA02 + NoSNE + NoPrompt  (基准-小尺寸)
  A2: 512 + EVA02 + NoSNE + NoPrompt  (基准-大尺寸)
  B2: 224 + DenseVLM + NoSNE + NoPrompt  (预训练对比)
  C2: 224 + DenseVLM + SNE + NoPrompt  (+SNE)
  D2: 224 + DenseVLM + SNE + Prompt  (+Prompt)
  E1: 512 + DenseVLM + SNE + Prompt  (完整模型)

Usage:
  python run_ablation.py --gpu 0                # 按顺序运行所有实验
  python run_ablation.py --gpu 0 --exp A1       # 运行单个实验
  python run_ablation.py --gpu 0 --exp A1,B2    # 运行指定实验
  python run_ablation.py --list                 # 列出所有实验
  python run_ablation.py --test                 # 仅测试 (使用已有权重)
"""

import argparse
import os
import subprocess
import time
from pathlib import Path

# 实验配置
EXPERIMENTS = {
    'A1': {
        'config': 'configs/ablation/exp_224_eva02_nosne_noprompt.py',
        'desc': '224 + EVA02 + NoSNE + NoPrompt (基准-小尺寸)',
        'compare': None,
    },
    'A2': {
        'config': 'configs/ablation/exp_512_eva02_nosne_noprompt.py', 
        'desc': '512 + EVA02 + NoSNE + NoPrompt (基准-大尺寸)',
        'compare': 'A1',  # 对比尺寸影响
    },
    'B2': {
        'config': 'configs/ablation/exp_224_densevlm_nosne_noprompt.py',
        'desc': '224 + DenseVLM + NoSNE + NoPrompt (预训练对比)',
        'compare': 'A1',  # 对比预训练权重
    },
    'C2': {
        'config': 'configs/ablation/exp_224_densevlm_sne_noprompt.py',
        'desc': '224 + DenseVLM + SNE(backbone-proj) + NoPrompt (+SNE)',
        'compare': 'B2',  # 对比 SNE 效果
    },
    'D2': {
        'config': 'configs/ablation/exp_224_densevlm_sne_prompt.py',
        'desc': '224 + DenseVLM + SNE + Prompt (+Prompt)',
        'compare': 'C2',  # 对比 Prompt 效果
    },
    'E1': {
        'config': 'configs/ablation/exp_512_densevlm_sne_prompt.py',
        'desc': '512 + DenseVLM + SNE + Prompt (完整模型)',
        'compare': 'D2',  # 对比尺寸提升
    },
}


def print_experiments():
    """打印所有实验配置"""
    print("\n" + "="*80)
    print("TQT 消融实验列表")
    print("="*80)
    
    # 表格格式
    print(f"\n{'实验':^4} | {'尺寸':^5} | {'权重':^10} | {'SNE':^5} | {'Prompt':^7} | 对比")
    print("-"*70)
    
    configs = [
        ('A1', '224', 'EVA02', '❌', '❌', '-'),
        ('A2', '512', 'EVA02', '❌', '❌', 'A1 (尺寸影响)'),
        ('B2', '224', 'DenseVLM', '❌', '❌', 'A1 (权重影响)'),
        ('C2', '224', 'DenseVLM', '✅ bkb-proj', '❌', 'B2 (SNE效果)'),
        ('D2', '224', 'DenseVLM', '✅', '✅', 'C2 (Prompt效果)'),
        ('E1', '512', 'DenseVLM', '✅', '✅', 'D2 (完整模型)'),
    ]
    
    for exp_id, size, weight, sne, prompt, compare in configs:
        print(f" {exp_id:^4} | {size:^5} | {weight:^10} | {sne:^5} | {prompt:^7} | {compare}")
    
    print("\n" + "="*80)
    print("对比分析:")
    print("-"*80)
    print("  1. 尺寸影响:    A1 vs A2  (224 vs 512, 其他相同)")
    print("  2. 权重影响:    A1 vs B2  (EVA02 vs DenseVLM)")
    print("  3. SNE 效果:    B2 vs C2  (无SNE vs 有SNE)")
    print("  4. Prompt效果:  C2 vs D2  (无Prompt vs 有Prompt)")
    print("  5. 完整对比:    A1 vs E1  (最简 vs 完整)")
    print("="*80 + "\n")


def run_experiment(exp_id, gpu_id, seed=42, deterministic=True, test_only=False):
    """运行单个实验"""
    exp = EXPERIMENTS.get(exp_id)
    if not exp:
        print(f"错误: 未知实验 {exp_id}")
        return False
    
    config_path = exp['config']
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在 {config_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"运行实验 {exp_id}: {exp['desc']}")
    print(f"配置文件: {config_path}")
    print(f"GPU: {gpu_id}, Seed: {seed}")
    print(f"{'='*60}\n")
    
    # 构建命令
    if test_only:
        # TODO: 需要指定 checkpoint 路径
        cmd = [
            'python', 'test.py',
            '--config', config_path,
            '--gpu-id', str(gpu_id),
        ]
    else:
        cmd = [
            'python', 'train.py',
            '--config', config_path,
            '--gpu-id', str(gpu_id),
            '--seed', str(seed),
        ]
        if deterministic:
            cmd.append('--deterministic')
    
    print(f"执行命令: {' '.join(cmd)}")
    
    # 运行
    start_time = time.time()
    result = subprocess.run(cmd, cwd='/root/tqdm')
    elapsed = time.time() - start_time
    
    print(f"\n实验 {exp_id} {'成功' if result.returncode == 0 else '失败'}")
    print(f"耗时: {elapsed/60:.1f} 分钟")
    
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='TQT 消融实验运行脚本')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--exp', type=str, default=None, 
                        help='要运行的实验ID (逗号分隔), 默认运行所有')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--list', action='store_true', help='列出所有实验')
    parser.add_argument('--test', action='store_true', help='仅测试 (需要先训练)')
    parser.add_argument('--no-deterministic', action='store_true', 
                        help='不使用确定性模式 (更快)')
    
    args = parser.parse_args()
    
    if args.list:
        print_experiments()
        return
    
    # 确定要运行的实验
    if args.exp:
        exp_ids = [e.strip().upper() for e in args.exp.split(',')]
        # 验证
        for exp_id in exp_ids:
            if exp_id not in EXPERIMENTS:
                print(f"错误: 未知实验 {exp_id}")
                print(f"可用实验: {', '.join(EXPERIMENTS.keys())}")
                return
    else:
        # 默认运行顺序
        exp_ids = ['A1', 'A2', 'B2', 'C2', 'D2', 'E1']
    
    print_experiments()
    print(f"\n将运行以下实验: {', '.join(exp_ids)}")
    print(f"GPU: {args.gpu}, Seed: {args.seed}")
    
    # 运行实验
    results = {}
    for exp_id in exp_ids:
        success = run_experiment(
            exp_id, 
            args.gpu, 
            seed=args.seed,
            deterministic=not args.no_deterministic,
            test_only=args.test
        )
        results[exp_id] = success
    
    # 汇总
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    for exp_id, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"  {exp_id}: {status}")
    print("="*60)


if __name__ == '__main__':
    main()
