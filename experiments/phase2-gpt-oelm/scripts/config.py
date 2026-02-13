#!/usr/bin/env python3
"""
配置文件
========
包含不同规模的模型配置和训练配置
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class ModelConfig:
    """模型配置"""
    vocab_size: int = 50257          # GPT-2词汇表大小
    d_model: int = 512               # 模型维度
    n_layers: int = 8                # Transformer层数
    n_heads: int = 8                 # 注意力头数
    d_head: int = 64                 # 每个头的维度
    d_ff: int = 2048                 # 前馈网络维度
    max_seq_len: int = 512           # 最大序列长度
    dropout: float = 0.1             # Dropout率
    
    # 正交随机注意力特有参数
    use_orthogonal_attention: bool = True  # 是否使用正交随机注意力
    num_orthogonal_features: int = 256     # 正交特征数量
    orthogonal_init_scale: float = 1.0     # 正交初始化缩放
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass  
class OptimizerConfig:
    """优化器配置"""
    learning_rate: float = 3e-4      # 学习率
    min_lr: float = 3e-5             # 最小学习率
    weight_decay: float = 0.1        # 权重衰减
    beta1: float = 0.9               # Adam beta1
    beta2: float = 0.95              # Adam beta2
    grad_clip: float = 1.0           # 梯度裁剪阈值
    
    # 学习率调度
    warmup_iters: int = 2000         # 预热步数
    lr_decay_iters: int = 100000     # 学习率衰减步数


@dataclass
class TrainingConfig:
    """训练配置"""
    # 训练参数
    batch_size: int = 32             # 每GPU的batch size
    max_iters: int = 100000          # 最大训练步数
    
    # 评估和保存
    eval_interval: int = 1000        # 评估间隔
    save_interval: int = 5000        # 保存间隔
    log_interval: int = 10           # 日志记录间隔
    
    # 数据参数
    dataset: str = "tinystories"     # 数据集名称
    data_dir: str = "data"           # 数据目录
    num_workers: int = 4             # 数据加载worker数
    
    # 系统参数
    device: str = "cuda"             # 设备
    dtype: str = "bfloat16"          # 数据类型
    compile: bool = True             # 是否使用torch.compile
    
    # 分布式参数
    backend: str = "nccl"            # 分布式后端
    
    # 检查点和日志
    out_dir: str = "out"             # 输出目录
    resume: str = ""                 # 恢复检查点路径
    wandb_project: str = "orthogonal-attention"  # WandB项目名
    wandb_run_name: str = ""         # WandB运行名


# =============================================================================
# 预定义配置
# =============================================================================

# 小型模型配置 (约 44M 参数)
def get_small_config() -> Dict[str, Any]:
    """获取小型模型配置"""
    return {
        'model': ModelConfig(
            d_model=384,
            n_layers=6,
            n_heads=6,
            d_head=64,
            d_ff=1536,
            max_seq_len=512,
            dropout=0.1,
            use_orthogonal_attention=True,
            num_orthogonal_features=192,
        ),
        'optimizer': OptimizerConfig(
            learning_rate=5e-4,
            min_lr=5e-5,
            warmup_iters=1000,
            lr_decay_iters=50000,
        ),
        'training': TrainingConfig(
            batch_size=64,
            max_iters=50000,
            eval_interval=500,
            save_interval=2500,
        )
    }


# 中型模型配置 (约 125M 参数)
def get_medium_config() -> Dict[str, Any]:
    """获取中型模型配置"""
    return {
        'model': ModelConfig(
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_head=64,
            d_ff=3072,
            max_seq_len=512,
            dropout=0.1,
            use_orthogonal_attention=True,
            num_orthogonal_features=384,
        ),
        'optimizer': OptimizerConfig(
            learning_rate=3e-4,
            min_lr=3e-5,
            warmup_iters=2000,
            lr_decay_iters=100000,
        ),
        'training': TrainingConfig(
            batch_size=32,
            max_iters=100000,
            eval_interval=1000,
            save_interval=5000,
        )
    }


# 大型模型配置 (约 354M 参数)
def get_large_config() -> Dict[str, Any]:
    """获取大型模型配置"""
    return {
        'model': ModelConfig(
            d_model=1024,
            n_layers=24,
            n_heads=16,
            d_head=64,
            d_ff=4096,
            max_seq_len=1024,
            dropout=0.1,
            use_orthogonal_attention=True,
            num_orthogonal_features=512,
        ),
        'optimizer': OptimizerConfig(
            learning_rate=2e-4,
            min_lr=2e-5,
            warmup_iters=4000,
            lr_decay_iters=200000,
        ),
        'training': TrainingConfig(
            batch_size=16,
            max_iters=200000,
            eval_interval=2000,
            save_interval=10000,
        )
    }


# 快速测试配置
def get_tiny_config() -> Dict[str, Any]:
    """获取快速测试配置"""
    return {
        'model': ModelConfig(
            d_model=128,
            n_layers=4,
            n_heads=4,
            d_head=32,
            d_ff=512,
            max_seq_len=256,
            dropout=0.1,
            use_orthogonal_attention=True,
            num_orthogonal_features=64,
        ),
        'optimizer': OptimizerConfig(
            learning_rate=1e-3,
            min_lr=1e-4,
            warmup_iters=100,
            lr_decay_iters=5000,
        ),
        'training': TrainingConfig(
            batch_size=128,
            max_iters=5000,
            eval_interval=500,
            save_interval=1000,
            log_interval=5,
        )
    }


# 标准注意力对比配置 (与正交随机注意力相同规模)
def get_standard_attention_config() -> Dict[str, Any]:
    """获取标准注意力对比配置"""
    config = get_medium_config()
    config['model'].use_orthogonal_attention = False
    return config


# 配置映射
CONFIGS = {
    'tiny': get_tiny_config,
    'small': get_small_config,
    'medium': get_medium_config,
    'large': get_large_config,
    'standard': get_standard_attention_config,
}


# =============================================================================
# Phase 3: GPT OELM Ablation Experiments Configuration
# =============================================================================

EXPERIMENT_CONFIGS = {
    # TinyStories Experiments
    'GPT-01': {
        'dataset': 'tinystories',
        'model_type': 'baseline',
        'lr': 3e-4,
        'max_steps': 100000,
        'description': 'TinyStories Baseline'
    },
    'GPT-02': {
        'dataset': 'tinystories',
        'model_type': 'oelm_v2',
        'lr': 1e-3,
        'max_steps': 100000,
        'description': 'TinyStories OELM-Freeze (orthogonal init)'
    },
    'GPT-03': {
        'dataset': 'tinystories',
        'model_type': 'oelm_random',
        'lr': 1e-3,
        'max_steps': 100000,
        'description': 'TinyStories OELM-Random (normal init) - Ablation'
    },
    # OpenWebText Experiments
    'GPT-04': {
        'dataset': 'openwebtext',
        'model_type': 'baseline',
        'lr': 3e-4,
        'max_steps': 150000,
        'description': 'OpenWebText Baseline'
    },
    'GPT-05': {
        'dataset': 'openwebtext',
        'model_type': 'oelm_v2',
        'lr': 1e-3,
        'max_steps': 150000,
        'description': 'OpenWebText OELM-Freeze'
    },
    # WikiText-103 Experiments
    'GPT-06': {
        'dataset': 'wikitext103',
        'model_type': 'baseline',
        'lr': 3e-4,
        'max_steps': 200000,
        'description': 'WikiText-103 Baseline'
    },
    'GPT-07': {
        'dataset': 'wikitext103',
        'model_type': 'oelm_v2',
        'lr': 1e-3,
        'max_steps': 200000,
        'description': 'WikiText-103 OELM-Freeze'
    },
}

# Model architecture for Phase 3 (Medium-512)
PHASE3_MODEL_CONFIG = {
    'vocab_size': 50257,
    'd_model': 512,
    'num_layers': 6,
    'num_heads': 8,
    'd_ff': 2048,
    'max_seq_len': 512,
    'dropout': 0.1,
}

# Training defaults for Phase 3
PHASE3_TRAINING_DEFAULTS = {
    'batch_size': 8,
    'warmup_steps': 2000,
    'min_lr': 3e-5,
    'weight_decay': 0.1,
    'beta1': 0.9,
    'beta2': 0.95,
    'grad_clip': 1.0,
    'val_interval': 1000,
    'save_interval': 5000,
}


def get_config(name: str) -> Dict[str, Any]:
    """
    获取指定名称的配置
    
    Args:
        name: 配置名称 ('tiny', 'small', 'medium', 'large', 'standard')
    
    Returns:
        配置字典
    """
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]()


# =============================================================================
# 配置合并工具
# =============================================================================

def merge_configs(
    model_config: ModelConfig,
    optimizer_config: OptimizerConfig,
    training_config: TrainingConfig
) -> Dict[str, Any]:
    """
    合并配置为训练脚本可用的格式
    
    Args:
        model_config: 模型配置
        optimizer_config: 优化器配置
        training_config: 训练配置
    
    Returns:
        合并后的配置字典
    """
    merged = {}
    merged.update(model_config.to_dict())
    merged.update(optimizer_config.to_dict())
    merged.update(training_config.to_dict())
    return merged


if __name__ == '__main__':
    # 打印所有可用配置
    print("Available configurations:")
    for name in CONFIGS:
        config = get_config(name)
        model = config['model']
        
        # 计算参数量
        vocab_params = model.vocab_size * model.d_model
        pos_params = model.max_seq_len * model.d_model
        
        # Transformer块参数量
        attn_params = model.n_layers * (
            4 * model.d_model * model.n_heads * model.d_head +  # Q, K, V, O
            model.n_heads * model.d_head * model.num_orthogonal_features * 2  # 正交特征
        )
        ffn_params = model.n_layers * (3 * model.d_model * model.d_ff)  # fc1, fc2, gate
        ln_params = model.n_layers * 4 * model.d_model  # 两个LayerNorm per block
        
        total_params = vocab_params + pos_params + attn_params + ffn_params + ln_params
        
        print(f"\n{name}:")
        print(f"  d_model: {model.d_model}")
        print(f"  n_layers: {model.n_layers}")
        print(f"  n_heads: {model.n_heads}")
        print(f"  max_seq_len: {model.max_seq_len}")
        print(f"  use_orthogonal_attention: {model.use_orthogonal_attention}")
        print(f"  estimated params: {total_params / 1e6:.1f}M")
