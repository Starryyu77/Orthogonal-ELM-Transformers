#!/usr/bin/env python3
"""
Q/K冻结机制对比实验控制脚本

管理三组实验：
- Group A: GPT-Base (标准Transformer)
- Group B: OELM-NoFreeze (正交初始化，不冻结Q/K)
- Group C: OELM-Freeze (正交初始化，冻结Q/K)

使用方法：
    # 顺序运行所有实验
    python scripts/experiment_qk_freeze.py --mode sequential

    # 并行运行 (需要足够GPU)
    python scripts/experiment_qk_freeze.py --mode parallel

    # 只运行特定组
    python scripts/experiment_qk_freeze.py --groups gpt_base,oelm_freeze

    # 使用MLDA服务器
    python scripts/experiment_qk_freeze.py --remote --host gpu43.dynip.ntu.edu.sg
"""

import os
import sys
import argparse
import subprocess
import json
import time
import signal
import threading
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """实验配置"""
    id: str
    name: str
    model_type: str
    freeze_qk: Optional[bool]
    gpus: str
    port: int
    out_dir: str
    wandb_name: str
    priority: int = 0  # 运行优先级，数字小的先运行


# 实验配置定义
EXPERIMENTS = {
    'gpt_base': ExperimentConfig(
        id='gpt_base',
        name='Group A: GPT-Base',
        model_type='gpt',
        freeze_qk=None,  # GPT不使用此参数
        gpus='0,1',
        port=29500,
        out_dir='models/checkpoints/exp_gpt_base',
        wandb_name='exp_gpt_base',
        priority=1
    ),
    'oelm_no_freeze': ExperimentConfig(
        id='oelm_no_freeze',
        name='Group B: OELM-NoFreeze',
        model_type='oelm',
        freeze_qk=False,
        gpus='2,3',
        port=29501,
        out_dir='models/checkpoints/exp_oelm_no_freeze',
        wandb_name='exp_oelm_no_freeze',
        priority=2
    ),
    'oelm_freeze': ExperimentConfig(
        id='oelm_freeze',
        name='Group C: OELM-Freeze',
        model_type='oelm',
        freeze_qk=True,
        gpus='0,1,2,3',  # 可以使用所有GPU
        port=29502,
        out_dir='models/checkpoints/exp_oelm_freeze',
        wandb_name='exp_oelm_freeze',
        priority=3
    )
}

# 通用训练参数
DEFAULT_TRAIN_ARGS = {
    'vocab_size': 50257,
    'd_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'd_ff': 2048,
    'seq_len': 512,
    'batch_size': 8,
    'max_steps': 100000,
    'warmup_steps': 2000,
    'max_lr': 3e-4,
    'min_lr': 3e-5,
    'weight_decay': 0.1,
    'grad_clip': 1.0,
    'beta1': 0.9,
    'beta2': 0.95,
    'data_path': 'data/tiny_stories/train.bin',
    'val_data_path': 'data/tiny_stories/val.bin',
    'log_interval': 100,
    'val_interval': 1000,
    'val_batches': 100,
    'save_interval': 5000,
    'use_wandb': False,
    'wandb_project': 'oelm-qk-freeze-exp',
    'seed': 42,
    'num_workers': 4,
}


class ExperimentRunner:
    """实验运行器"""

    def __init__(
        self,
        experiments: List[str],
        mode: str = 'sequential',
        remote_host: Optional[str] = None,
        remote_user: Optional[str] = None,
        dry_run: bool = False
    ):
        self.experiments = experiments
        self.mode = mode
        self.remote_host = remote_host
        self.remote_user = remote_user
        self.dry_run = dry_run
        self.processes: Dict[str, subprocess.Popen] = {}
        self.results: Dict[str, dict] = {}
        self.start_time: Optional[datetime] = None

        # 信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """处理中断信号"""
        logger.warning(f"\n收到信号 {signum}，正在停止所有实验...")
        self.stop_all()
        sys.exit(1)

    def build_command(self, exp_config: ExperimentConfig) -> List[str]:
        """构建训练命令"""
        cmd = ['python', '-m', 'torch.distributed.run']
        cmd.extend(['--nproc_per_node', str(len(exp_config.gpus.split(',')))])
        cmd.extend(['--master_port', str(exp_config.port)])
        cmd.append('scripts/02-训练脚本/train.py')

        # 模型类型
        cmd.extend(['--model_type', exp_config.model_type])

        # freeze_qk参数（仅OELM）
        if exp_config.freeze_qk is not None:
            cmd.extend(['--freeze_qk', str(exp_config.freeze_qk).lower()])

        # 通用参数
        for key, value in DEFAULT_TRAIN_ARGS.items():
            if key == 'data_path' or key == 'val_data_path':
                # 确保路径存在
                if not Path(value).exists():
                    logger.warning(f"数据文件不存在: {value}")
            cmd.extend([f'--{key.replace("_", "-")}', str(value)])

        # 输出目录和WandB名称
        cmd.extend(['--out-dir', exp_config.out_dir])
        if exp_config.wandb_name:
            cmd.extend(['--wandb-run-name', exp_config.wandb_name])

        return cmd

    def run_experiment(
        self,
        exp_id: str,
        exp_config: ExperimentConfig
    ) -> dict:
        """运行单个实验"""
        logger.info(f"\n{'='*70}")
        logger.info(f"启动实验: {exp_config.name}")
        logger.info(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"{'='*70}")

        # 创建输出目录
        Path(exp_config.out_dir).mkdir(parents=True, exist_ok=True)

        # 构建环境变量
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = exp_config.gpus

        # 构建命令
        cmd = self.build_command(exp_config)
        logger.info(f"命令: {' '.join(cmd)}")
        logger.info(f"GPU: {exp_config.gpus}")
        logger.info(f"输出目录: {exp_config.out_dir}")

        if self.dry_run:
            logger.info("[DRY RUN] 不实际执行训练")
            return {'status': 'dry_run', 'exp_id': exp_id}

        # 启动训练进程
        log_file = Path(exp_config.out_dir) / 'training.log'
        with open(log_file, 'w') as f:
            f.write(f"Experiment: {exp_config.name}\n")
            f.write(f"Started: {datetime.now().isoformat()}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"{'='*70}\n\n")

        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        self.processes[exp_id] = process

        # 实时读取输出
        start_time = time.time()
        with open(log_file, 'a') as f:
            for line in process.stdout:
                f.write(line)
                f.flush()
                # 打印到控制台（可选）
                if 'Step' in line or 'Validation' in line or 'Saved' in line:
                    logger.info(f"[{exp_id}] {line.strip()}")

        # 等待完成
        return_code = process.wait()
        elapsed = time.time() - start_time

        # 记录结果
        result = {
            'exp_id': exp_id,
            'name': exp_config.name,
            'status': 'success' if return_code == 0 else 'failed',
            'return_code': return_code,
            'elapsed_time': elapsed,
            'elapsed_hours': elapsed / 3600,
            'completed_at': datetime.now().isoformat(),
            'out_dir': exp_config.out_dir,
            'log_file': str(log_file)
        }

        # 保存结果
        result_file = Path(exp_config.out_dir) / 'experiment_result.json'
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"\n实验 {exp_id} 完成")
        logger.info(f"状态: {result['status']}")
        logger.info(f"耗时: {result['elapsed_hours']:.2f} 小时")

        return result

    def run_remote(
        self,
        exp_id: str,
        exp_config: ExperimentConfig
    ) -> dict:
        """在远程服务器上运行实验"""
        logger.info(f"\n{'='*70}")
        logger.info(f"远程启动实验: {exp_config.name}")
        logger.info(f"主机: {self.remote_host}")
        logger.info(f"{'='*70}")

        # 构建远程命令
        remote_cmd = f"cd ~/Orthogonal_ELM_Transformers/Train && "
        remote_cmd += f"export CUDA_VISIBLE_DEVICES={exp_config.gpus} && "

        # 构建训练命令
        train_cmd = ' '.join(self.build_command(exp_config))
        remote_cmd += train_cmd

        # 构建SSH命令
        ssh_cmd = [
            'ssh',
            f'{self.remote_user}@{self.remote_host}',
            remote_cmd
        ]

        logger.info(f"远程命令: {remote_cmd}")

        if self.dry_run:
            logger.info("[DRY RUN] 不实际执行")
            return {'status': 'dry_run', 'exp_id': exp_id}

        # 启动远程进程
        process = subprocess.Popen(
            ssh_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        self.processes[exp_id] = process

        # 读取输出
        start_time = time.time()
        for line in process.stdout:
            logger.info(f"[{exp_id}] {line.strip()}")

        return_code = process.wait()
        elapsed = time.time() - start_time

        result = {
            'exp_id': exp_id,
            'name': exp_config.name,
            'status': 'success' if return_code == 0 else 'failed',
            'return_code': return_code,
            'elapsed_time': elapsed,
            'elapsed_hours': elapsed / 3600,
            'completed_at': datetime.now().isoformat()
        }

        logger.info(f"\n远程实验 {exp_id} 完成")
        logger.info(f"状态: {result['status']}")

        return result

    def run_sequential(self) -> Dict[str, dict]:
        """顺序运行所有实验"""
        logger.info("\n" + "="*70)
        logger.info("顺序运行模式")
        logger.info("="*70)

        # 按优先级排序
        sorted_exps = sorted(
            self.experiments,
            key=lambda x: EXPERIMENTS[x].priority
        )

        for exp_id in sorted_exps:
            exp_config = EXPERIMENTS[exp_id]

            if self.remote_host:
                result = self.run_remote(exp_id, exp_config)
            else:
                result = self.run_experiment(exp_id, exp_config)

            self.results[exp_id] = result

            # 如果实验失败，询问是否继续
            if result['status'] == 'failed':
                logger.error(f"实验 {exp_id} 失败!")
                # 可以选择在这里添加中断逻辑

        return self.results

    def run_parallel(self) -> Dict[str, dict]:
        """并行运行所有实验"""
        logger.info("\n" + "="*70)
        logger.info("并行运行模式")
        logger.info("警告: 确保有足够的GPU资源!")
        logger.info("="*70)

        threads = []

        for exp_id in self.experiments:
            exp_config = EXPERIMENTS[exp_id]

            if self.remote_host:
                target = self.run_remote
            else:
                target = self.run_experiment

            thread = threading.Thread(
                target=lambda eid=exp_id, ecfg=exp_config: self._thread_wrapper(eid, ecfg, target)
            )
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        return self.results

    def _thread_wrapper(self, exp_id: str, exp_config: ExperimentConfig, target):
        """线程包装器"""
        try:
            result = target(exp_id, exp_config)
            self.results[exp_id] = result
        except Exception as e:
            logger.error(f"实验 {exp_id} 异常: {e}")
            self.results[exp_id] = {
                'exp_id': exp_id,
                'status': 'error',
                'error': str(e)
            }

    def stop_all(self):
        """停止所有运行中的实验"""
        logger.info("停止所有实验...")
        for exp_id, process in self.processes.items():
            if process.poll() is None:  # 仍在运行
                logger.info(f"停止 {exp_id}...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()

    def generate_summary(self):
        """生成实验总结"""
        summary_file = Path('experiments/qk_freeze_summary.json')
        summary_file.parent.mkdir(exist_ok=True)

        summary = {
            'experiment_name': 'Q/K Freeze Mechanism Comparison',
            'started_at': self.start_time.isoformat() if self.start_time else None,
            'completed_at': datetime.now().isoformat(),
            'mode': self.mode,
            'results': self.results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\n总结已保存: {summary_file}")

        # 打印表格
        logger.info("\n" + "="*70)
        logger.info("实验结果汇总")
        logger.info("="*70)
        logger.info(f"{'Experiment':<20} {'Status':<10} {'Time (h)':<10}")
        logger.info("-"*70)

        for exp_id, result in self.results.items():
            name = EXPERIMENTS[exp_id].name.split(':')[0]
            status = result.get('status', 'unknown')
            elapsed = result.get('elapsed_hours', 0)
            logger.info(f"{name:<20} {status:<10} {elapsed:<10.2f}")

        logger.info("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Q/K Freeze Mechanism Experiment Controller'
    )

    parser.add_argument(
        '--groups',
        type=str,
        default='all',
        help='要运行的实验组，逗号分隔 (gpt_base,oelm_no_freeze,oelm_freeze) 或 all'
    )

    parser.add_argument(
        '--mode',
        type=str,
        default='sequential',
        choices=['sequential', 'parallel'],
        help='运行模式: sequential (顺序) 或 parallel (并行)'
    )

    parser.add_argument(
        '--remote',
        action='store_true',
        help='在远程服务器上运行'
    )

    parser.add_argument(
        '--host',
        type=str,
        default='gpu43.dynip.ntu.edu.sg',
        help='远程服务器地址'
    )

    parser.add_argument(
        '--user',
        type=str,
        default='s125mdg43_10',
        help='远程服务器用户名'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='干运行模式（只打印命令，不实际执行）'
    )

    args = parser.parse_args()

    # 确定要运行的实验
    if args.groups == 'all':
        experiments = list(EXPERIMENTS.keys())
    else:
        experiments = [g.strip() for g in args.groups.split(',')]
        # 验证实验ID
        for exp_id in experiments:
            if exp_id not in EXPERIMENTS:
                logger.error(f"未知实验ID: {exp_id}")
                logger.error(f"可用选项: {', '.join(EXPERIMENTS.keys())}")
                return 1

    logger.info("="*70)
    logger.info("Q/K矩阵冻结机制对比实验")
    logger.info("="*70)
    logger.info(f"实验组: {', '.join(experiments)}")
    logger.info(f"运行模式: {args.mode}")
    if args.remote:
        logger.info(f"远程服务器: {args.user}@{args.host}")
    logger.info("="*70)

    # 创建运行器
    runner = ExperimentRunner(
        experiments=experiments,
        mode=args.mode,
        remote_host=args.host if args.remote else None,
        remote_user=args.user if args.remote else None,
        dry_run=args.dry_run
    )

    runner.start_time = datetime.now()

    # 运行实验
    try:
        if args.mode == 'sequential':
            results = runner.run_sequential()
        else:
            results = runner.run_parallel()
    except KeyboardInterrupt:
        logger.warning("\n用户中断，正在清理...")
        runner.stop_all()
        return 1

    # 生成总结
    runner.generate_summary()

    # 返回状态码
    failed = sum(1 for r in results.values() if r.get('status') != 'success')
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
