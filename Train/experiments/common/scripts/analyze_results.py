#!/usr/bin/env python3
"""
实验结果分析脚本

功能:
1. 加载和对比多个实验的 timing_stats.json
2. 生成对比表格
3. 计算性能差距

用法:
    python analyze_results.py --exp1 outputs/GPT-01_baseline --exp2 outputs/GPT-02_oelm_freeze
    python analyze_results.py --all                                    # 分析所有实验
"""

import json
import argparse
import glob
from pathlib import Path
from typing import Dict, List, Optional


def load_timing_stats(exp_dir: str) -> Optional[Dict]:
    """加载实验的 timing_stats.json"""
    stats_file = Path(exp_dir) / "timing_stats.json"
    if not stats_file.exists():
        return None

    with open(stats_file) as f:
        return json.load(f)


def format_time(seconds: float) -> str:
    """格式化时间"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def compare_experiments(exp1_dir: str, exp2_dir: str):
    """对比两个实验"""
    stats1 = load_timing_stats(exp1_dir)
    stats2 = load_timing_stats(exp2_dir)

    if not stats1 or not stats2:
        print("错误: 无法加载实验数据")
        return

    print("=" * 70)
    print(f"实验对比: {stats1.get('model_type', 'N/A')} vs {stats2.get('model_type', 'N/A')}")
    print("=" * 70)
    print()

    # 基本信息
    print("基本信息:")
    print(f"  数据集: {stats1.get('dataset', 'N/A')}")
    print(f"  实验1: {Path(exp1_dir).name}")
    print(f"  实验2: {Path(exp2_dir).name}")
    print()

    # 性能对比
    print("性能对比:")
    ppl1 = stats1.get('final_perplexity', 0)
    ppl2 = stats2.get('final_perplexity', 0)
    ppl_diff = ((ppl2 - ppl1) / ppl1 * 100) if ppl1 > 0 else 0

    print(f"  Final PPL:")
    print(f"    实验1: {ppl1:.2f}")
    print(f"    实验2: {ppl2:.2f}")
    print(f"    差距:  {ppl_diff:+.2f}%")
    print()

    # 训练时间对比
    print("训练时间对比:")
    time1 = stats1.get('total_wall_time', 0)
    time2 = stats2.get('total_wall_time', 0)
    time_diff = ((time2 - time1) / time1 * 100) if time1 > 0 else 0

    print(f"  实验1: {format_time(time1)}")
    print(f"  实验2: {format_time(time2)}")
    print(f"  差距:  {time_diff:+.2f}%")
    print()

    # 步时间对比
    print("步时间对比:")
    step1 = stats1.get('mean_step_time', 0) * 1000  # 转换为ms
    step2 = stats2.get('mean_step_time', 0) * 1000
    step_diff = ((step2 - step1) / step1 * 100) if step1 > 0 else 0

    print(f"  实验1: {step1:.2f} ms/step")
    print(f"  实验2: {step2:.2f} ms/step")
    print(f"  差距:  {step_diff:+.2f}%")
    print()


def analyze_all_experiments(base_dir: str):
    """分析所有实验"""
    output_dir = Path(base_dir)
    if not output_dir.exists():
        print(f"错误: 目录不存在 {base_dir}")
        return

    # 查找所有 timing_stats.json
    stats_files = list(output_dir.glob("**/timing_stats.json"))

    if not stats_files:
        print("未找到实验结果")
        return

    print("=" * 90)
    print("所有实验结果汇总")
    print("=" * 90)
    print()

    # 表头
    print(f"{'实验ID':<30} {'数据集':<15} {'模型':<15} {'PPL':<10} {'训练时间':<15} {'步时间(ms)':<12}")
    print("-" * 90)

    results = []
    for stats_file in sorted(stats_files):
        with open(stats_file) as f:
            stats = json.load(f)

        exp_name = stats_file.parent.name
        dataset = stats.get('dataset', 'N/A')
        model = stats.get('model_type', 'N/A')
        ppl = stats.get('final_perplexity', 0)
        time_str = stats.get('total_formatted', 'N/A')
        step_time = stats.get('mean_step_time', 0) * 1000

        results.append({
            'name': exp_name,
            'dataset': dataset,
            'model': model,
            'ppl': ppl,
            'time': time_str,
            'step_time': step_time
        })

    # 按数据集排序
    for r in sorted(results, key=lambda x: (x['dataset'], x['model'])):
        print(f"{r['name']:<30} {r['dataset']:<15} {r['model']:<15} {r['ppl']:<10.2f} {r['time']:<15} {r['step_time']:<12.2f}")

    print()


def main():
    parser = argparse.ArgumentParser(description='分析实验结果')
    parser.add_argument('--exp1', help='第一个实验目录')
    parser.add_argument('--exp2', help='第二个实验目录')
    parser.add_argument('--all', action='store_true', help='分析所有实验')
    parser.add_argument('--base-dir', default='./gpt-oelm-project/outputs',
                       help='实验输出基础目录')

    args = parser.parse_args()

    if args.all:
        analyze_all_experiments(args.base_dir)
    elif args.exp1 and args.exp2:
        compare_experiments(args.exp1, args.exp2)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
