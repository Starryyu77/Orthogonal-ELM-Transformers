#!/usr/bin/env python3
"""
Q/K冻结实验结果分析脚本

收集并分析三组实验的结果，生成对比报告和可视化

使用方法:
    python scripts/analyze_freeze_experiment.py
    python scripts/analyze_freeze_experiment.py --plot
    python scripts/analyze_freeze_experiment.py --report markdown
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """实验结果数据结构"""
    exp_id: str
    name: str
    final_val_loss: Optional[float] = None
    final_val_ppl: Optional[float] = None
    best_val_loss: Optional[float] = None
    best_val_ppl: Optional[float] = None
    total_params: int = 0
    trainable_params: int = 0
    frozen_params: int = 0
    training_hours: float = 0.0
    steps_per_second: float = 0.0
    status: str = 'unknown'


EXPERIMENT_DIRS = {
    'gpt_base': 'models/checkpoints/exp_gpt_base',
    'oelm_no_freeze': 'models/checkpoints/exp_oelm_no_freeze',
    'oelm_freeze': 'models/checkpoints/exp_oelm_freeze',
}


def parse_log_file(log_path: Path) -> Dict:
    """解析训练日志文件"""
    if not log_path.exists():
        return {}

    results = {
        'train_losses': [],
        'val_losses': [],
        'val_ppls': [],
        'steps': [],
        'val_steps': []
    }

    with open(log_path, 'r') as f:
        for line in f:
            # 解析训练步骤
            train_match = re.search(
                r'Step\s+(\d+).*Loss:\s+([\d.]+).*PPL:\s+([\d.]+)', line
            )
            if train_match:
                step = int(train_match.group(1))
                loss = float(train_match.group(2))
                ppl = float(train_match.group(3))
                results['steps'].append(step)
                results['train_losses'].append(loss)

            # 解析验证步骤
            val_match = re.search(
                r'Validation.*Loss:\s+([\d.]+).*PPL:\s+([\d.]+)', line
            )
            if val_match:
                loss = float(val_match.group(1))
                ppl = float(val_match.group(2))
                results['val_losses'].append(loss)
                results['val_ppls'].append(ppl)
                if results['steps']:
                    results['val_steps'].append(results['steps'][-1])

    return results


def load_checkpoint_info(checkpoint_dir: Path) -> Dict:
    """加载检查点信息"""
    info = {}

    # 尝试加载final.pt或best_model.pt
    for ckpt_name in ['final.pt', 'best_model.pt']:
        ckpt_path = checkpoint_dir / ckpt_name
        if ckpt_path.exists():
            try:
                import torch
                ckpt = torch.load(ckpt_path, map_location='cpu')
                info['checkpoint'] = ckpt_name
                info['step'] = ckpt.get('step', 0)
                info['best_val_loss'] = ckpt.get('best_val_loss', None)
                break
            except Exception as e:
                logger.warning(f"无法加载检查点 {ckpt_path}: {e}")

    return info


def count_parameters(exp_dir: Path) -> Tuple[int, int, int]:
    """统计模型参数"""
    total = 0
    trainable = 0
    frozen = 0

    # 从日志中解析
    log_file = exp_dir / 'training.log'
    if log_file.exists():
        with open(log_file, 'r') as f:
            content = f.read()

            # 查找参数统计
            total_match = re.search(r'Total parameters:\s+([\d,]+)', content)
            trainable_match = re.search(r'Trainable parameters:\s+([\d,]+)', content)
            frozen_match = re.search(r'Frozen parameters:\s+([\d,]+)', content)

            if total_match:
                total = int(total_match.group(1).replace(',', ''))
            if trainable_match:
                trainable = int(trainable_match.group(1).replace(',', ''))
            if frozen_match:
                frozen = int(frozen_match.group(1).replace(',', ''))

    return total, trainable, frozen


def analyze_experiment(exp_id: str) -> ExperimentResult:
    """分析单个实验"""
    exp_dir = Path(EXPERIMENT_DIRS.get(exp_id, f'models/checkpoints/{exp_id}'))

    result = ExperimentResult(
        exp_id=exp_id,
        name=exp_id.replace('_', ' ').title()
    )

    # 加载结果文件
    result_file = exp_dir / 'experiment_result.json'
    if result_file.exists():
        with open(result_file, 'r') as f:
            data = json.load(f)
            result.status = data.get('status', 'unknown')
            result.training_hours = data.get('elapsed_hours', 0.0)

    # 解析日志
    log_file = exp_dir / 'training.log'
    log_data = parse_log_file(log_file)

    if log_data.get('val_losses'):
        result.final_val_loss = log_data['val_losses'][-1]
        result.final_val_ppl = log_data['val_ppls'][-1]
        result.best_val_loss = min(log_data['val_losses'])
        result.best_val_ppl = min(log_data['val_ppls'])

    # 统计参数
    total, trainable, frozen = count_parameters(exp_dir)
    result.total_params = total
    result.trainable_params = trainable
    result.frozen_params = frozen

    return result


def generate_comparison_table(results: List[ExperimentResult]) -> str:
    """生成对比表格"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("Q/K冻结实验结果对比")
    lines.append("="*80)
    lines.append("")

    # 表头
    header = f"{'Experiment':<20} {'Status':<10} {'Val PPL':<10} {'Best PPL':<10} {'Params':<12} {'Frozen %':<10} {'Hours':<8}"
    lines.append(header)
    lines.append("-"*80)

    # 数据行
    for r in results:
        name = r.exp_id.replace('_', ' ').title()
        status = r.status
        val_ppl = f"{r.final_val_ppl:.2f}" if r.final_val_ppl else "N/A"
        best_ppl = f"{r.best_val_ppl:.2f}" if r.best_val_ppl else "N/A"
        params = f"{r.total_params/1e6:.1f}M" if r.total_params else "N/A"
        frozen_pct = f"{100*r.frozen_params/r.total_params:.1f}%" if r.total_params else "N/A"
        hours = f"{r.training_hours:.1f}" if r.training_hours else "N/A"

        line = f"{name:<20} {status:<10} {val_ppl:<10} {best_ppl:<10} {params:<12} {frozen_pct:<10} {hours:<8}"
        lines.append(line)

    lines.append("="*80)

    return "\n".join(lines)


def generate_hypothesis_validation(results: List[ExperimentResult]) -> str:
    """生成假设验证报告"""
    lines = []
    lines.append("\n" + "="*80)
    lines.append("假设验证")
    lines.append("="*80)

    # 建立索引
    result_map = {r.exp_id: r for r in results}

    # H1: 参数减少15%
    lines.append("\nH1: OELM-Freeze比OELM-NoFreeze参数减少~15%")
    if 'oelm_no_freeze' in result_map and 'oelm_freeze' in result_map:
        no_freeze = result_map['oelm_no_freeze']
        freeze = result_map['oelm_freeze']

        if no_freeze.total_params and freeze.total_params:
            param_diff = no_freeze.trainable_params - freeze.trainable_params
            param_pct = 100 * param_diff / no_freeze.trainable_params
            lines.append(f"  NoFreeze可训练参数: {no_freeze.trainable_params:,}")
            lines.append(f"  Freeze可训练参数: {freeze.trainable_params:,}")
            lines.append(f"  减少: {param_diff:,} ({param_pct:.1f}%)")
            lines.append(f"  结果: {'✓ 通过' if 10 <= param_pct <= 20 else '✗ 未通过'}")

    # H2: 性能保持
    lines.append("\nH2: Freeze与NoFreeze的Val PPL差距 < 5%")
    if 'oelm_no_freeze' in result_map and 'oelm_freeze' in result_map:
        no_freeze = result_map['oelm_no_freeze']
        freeze = result_map['oelm_freeze']

        if no_freeze.best_val_ppl and freeze.best_val_ppl:
            ppl_diff_pct = 100 * abs(freeze.best_val_ppl - no_freeze.best_val_ppl) / no_freeze.best_val_ppl
            lines.append(f"  NoFreeze Best PPL: {no_freeze.best_val_ppl:.2f}")
            lines.append(f"  Freeze Best PPL: {freeze.best_val_ppl:.2f}")
            lines.append(f"  差距: {ppl_diff_pct:.2f}%")
            lines.append(f"  结果: {'✓ 通过' if ppl_diff_pct < 5 else '✗ 未通过'}")

    # H3: 速度提升
    lines.append("\nH3: Freeze训练速度 ≥ NoFreeze")
    if 'oelm_no_freeze' in result_map and 'oelm_freeze' in result_map:
        no_freeze = result_map['oelm_no_freeze']
        freeze = result_map['oelm_freeze']

        if no_freeze.training_hours and freeze.training_hours:
            lines.append(f"  NoFreeze耗时: {no_freeze.training_hours:.1f}h")
            lines.append(f"  Freeze耗时: {freeze.training_hours:.1f}h")
            speedup = no_freeze.training_hours / freeze.training_hours if freeze.training_hours > 0 else 0
            lines.append(f"  速度比: {speedup:.2f}x")
            lines.append(f"  结果: {'✓ 通过' if speedup >= 1.0 else '✗ 未通过'}")

    # H4: 竞争力
    lines.append("\nH4: Freeze性能接近GPT-Base (差距<10%)")
    if 'gpt_base' in result_map and 'oelm_freeze' in result_map:
        gpt = result_map['gpt_base']
        freeze = result_map['oelm_freeze']

        if gpt.best_val_ppl and freeze.best_val_ppl:
            ppl_diff_pct = 100 * (freeze.best_val_ppl - gpt.best_val_ppl) / gpt.best_val_ppl
            lines.append(f"  GPT-Base Best PPL: {gpt.best_val_ppl:.2f}")
            lines.append(f"  Freeze Best PPL: {freeze.best_val_ppl:.2f}")
            lines.append(f"  差距: {ppl_diff_pct:.2f}%")
            lines.append(f"  结果: {'✓ 通过' if ppl_diff_pct < 10 else '✗ 未通过'}")

    lines.append("\n" + "="*80)

    return "\n".join(lines)


def plot_results(results: List[ExperimentResult], output_dir: Path):
    """绘制结果图表"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib未安装，无法绘图")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 收集各实验的日志数据
    all_log_data = {}
    for r in results:
        log_file = Path(EXPERIMENT_DIRS.get(r.exp_id)) / 'training.log'
        all_log_data[r.exp_id] = parse_log_file(log_file)

    # 图1: 验证PPL对比
    plt.figure(figsize=(12, 6))
    for r in results:
        log_data = all_log_data.get(r.exp_id, {})
        if log_data.get('val_steps') and log_data.get('val_ppls'):
            label = r.exp_id.replace('_', ' ').title()
            plt.plot(log_data['val_steps'], log_data['val_ppls'], label=label, marker='o', markersize=3)

    plt.xlabel('Training Steps')
    plt.ylabel('Validation Perplexity')
    plt.title('Validation PPL Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(output_dir / 'val_ppl_comparison.png', dpi=150)
    plt.close()

    # 图2: 参数效率对比
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [r.exp_id.replace('_', '\n').title() for r in results]
    trainable = [r.trainable_params / 1e6 for r in results]
    frozen = [r.frozen_params / 1e6 for r in results]

    x = np.arange(len(names))
    width = 0.35

    ax.bar(x - width/2, trainable, width, label='Trainable', color='steelblue')
    ax.bar(x + width/2, frozen, width, label='Frozen', color='coral')

    ax.set_ylabel('Parameters (Millions)')
    ax.set_title('Parameter Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_distribution.png', dpi=150)
    plt.close()

    logger.info(f"图表已保存到: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Analyze Q/K Freeze Experiment Results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--output', type=str, default='experiments/analysis',
                        help='Output directory for analysis')
    parser.add_argument('--format', type=str, default='text', choices=['text', 'markdown', 'json'],
                        help='Output format')

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("Q/K冻结实验结果分析")
    logger.info("="*80)

    # 分析所有实验
    results = []
    for exp_id in EXPERIMENT_DIRS.keys():
        logger.info(f"\n分析实验: {exp_id}")
        result = analyze_experiment(exp_id)
        results.append(result)

    # 生成对比表格
    table = generate_comparison_table(results)
    print(table)

    # 保存表格
    with open(output_dir / 'comparison_table.txt', 'w') as f:
        f.write(table)

    # 生成假设验证
    validation = generate_hypothesis_validation(results)
    print(validation)

    with open(output_dir / 'hypothesis_validation.txt', 'w') as f:
        f.write(validation)

    # 生成图表
    if args.plot:
        logger.info("\n生成图表...")
        plot_results(results, output_dir)

    # 保存JSON结果
    results_dict = {
        'analysis_date': datetime.now().isoformat(),
        'experiments': [
            {
                'exp_id': r.exp_id,
                'name': r.name,
                'status': r.status,
                'final_val_loss': r.final_val_loss,
                'final_val_ppl': r.final_val_ppl,
                'best_val_loss': r.best_val_loss,
                'best_val_ppl': r.best_val_ppl,
                'total_params': r.total_params,
                'trainable_params': r.trainable_params,
                'frozen_params': r.frozen_params,
                'training_hours': r.training_hours
            }
            for r in results
        ]
    }

    with open(output_dir / 'analysis_results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"\n分析完成! 结果保存到: {output_dir}")

    return 0


if __name__ == '__main__':
    from datetime import datetime
    sys.exit(main())
