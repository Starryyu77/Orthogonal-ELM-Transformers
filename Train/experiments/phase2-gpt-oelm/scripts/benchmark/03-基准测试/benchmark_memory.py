#!/usr/bin/env python3
"""
显存测试脚本 (benchmark_memory.py)
===================================

用于测试正交随机注意力的显存占用情况。

功能:
1. 峰值显存占用测量
2. 显存分解分析
3. 不同批次的显存测试

用法:
    python benchmark_memory.py --attention orthogonal --batch_sizes 1 4 8 16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class MemoryConfig:
    """显存测试配置"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    vocab_size: int = 10000
    dropout: float = 0.1
    num_random_features: int = 256
    device: str = 'cuda'


# =============================================================================
# 注意力机制实现
# =============================================================================

class OrthogonalRandomFeatures(nn.Module):
    """正交随机特征"""
    def __init__(self, d_model: int, num_features: int):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features

        W = torch.randn(d_model, num_features)
        Q, _ = torch.linalg.qr(W)
        self.register_buffer('omega', Q[:, :num_features])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.matmul(x, self.omega.to(x.device))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class OrthogonalRandomAttention(nn.Module):
    """正交随机注意力"""
    def __init__(self, d_model: int, n_heads: int, num_features: int = 256, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.num_features = num_features

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.ortho_features = OrthogonalRandomFeatures(self.d_head, num_features // 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        Q_features = self.ortho_features(Q.transpose(1, 2).reshape(-1, seq_len, self.d_head))
        K_features = self.ortho_features(K.transpose(1, 2).reshape(-1, seq_len, self.d_head))

        Q_features = Q_features.view(batch_size, self.n_heads, seq_len, self.num_features)
        K_features = K_features.view(batch_size, self.n_heads, seq_len, self.num_features)

        KV = torch.matmul(K_features.transpose(-2, -1), V)
        Z = 1 / (torch.matmul(Q_features, K_features.transpose(-2, -1)).sum(dim=-1, keepdim=True) + 1e-6)
        out = torch.matmul(Q_features, KV) * Z

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class StandardMultiHeadAttention(nn.Module):
    """标准多头注意力"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_head ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_o(out)


class SynthesizerAttention(nn.Module):
    """Synthesizer注意力"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.synthetic_attn = nn.Linear(d_model, n_heads * d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        attn_weights = self.synthetic_attn(x)
        attn_weights = attn_weights.view(batch_size, seq_len, self.n_heads, seq_len)
        attn_weights = attn_weights.permute(0, 2, 1, 3)

        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        out = torch.matmul(attn_weights, V)

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1,
                 attention_type: str = 'standard', num_features: int = 256):
        super().__init__()

        if attention_type == 'standard':
            self.attention = StandardMultiHeadAttention(d_model, n_heads, dropout)
        elif attention_type == 'orthogonal':
            self.attention = OrthogonalRandomAttention(d_model, n_heads, num_features, dropout)
        elif attention_type == 'synthesizer':
            self.attention = SynthesizerAttention(d_model, n_heads, dropout)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout1(attn_out)

        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout2(ffn_out)

        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, config: MemoryConfig, attention_type: str = 'standard'):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        self.dropout = nn.Dropout(config.dropout)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, attention_type, config.num_random_features
            )
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# =============================================================================
# 显存测试器
# =============================================================================

class MemoryTester:
    """显存测试器"""

    def __init__(self, config: MemoryConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    def get_model_memory(self, model: nn.Module) -> Dict:
        """获取模型显存信息"""
        # 参数显存
        param_mem = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2

        # 缓冲区显存
        buffer_mem = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024**2

        # 按层分解
        layer_mem = {}
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子模块
                mem = sum(p.numel() * p.element_size() for p in module.parameters()) / 1024**2
                if mem > 0:
                    layer_mem[name] = mem

        return {
            'params_mb': param_mem,
            'buffers_mb': buffer_mem,
            'total_mb': param_mem + buffer_mem,
            'layer_breakdown': layer_mem,
        }

    def measure_forward_memory(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测量前向传播显存"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        start_mem = torch.cuda.memory_allocated() / 1024**2
        output = model(input_ids)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        end_mem = torch.cuda.memory_allocated() / 1024**2

        return {
            'start_mb': start_mem,
            'peak_mb': peak_mem,
            'end_mb': end_mem,
            'activation_mb': end_mem - start_mem,
        }

    def measure_backward_memory(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测量反向传播显存"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        start_mem = torch.cuda.memory_allocated() / 1024**2

        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, self.config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
        end_mem = torch.cuda.memory_allocated() / 1024**2

        return {
            'start_mb': start_mem,
            'peak_mb': peak_mem,
            'end_mb': end_mem,
            'gradients_mb': end_mem - start_mem,
        }

    def measure_optimizer_memory(self, model: nn.Module) -> Dict:
        """测量优化器状态显存"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        optimizer_mem = 0
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    optimizer_mem += value.numel() * value.element_size()

        optimizer_mem_mb = optimizer_mem / 1024**2

        return {
            'optimizer_state_mb': optimizer_mem_mb,
        }

    def measure_full_training_memory(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测量完整训练显存"""
        if not torch.cuda.is_available():
            return {'error': 'CUDA not available'}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        # 完整训练步骤
        optimizer.zero_grad()
        output = model(input_ids)
        loss = criterion(output.view(-1, self.config.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        return {
            'peak_memory_mb': peak_mem,
            'peak_memory_gb': peak_mem / 1024,
        }

    def test_batch_sizes(self, attention_type: str, seq_len: int, batch_sizes: List[int]) -> Dict:
        """测试不同批次的显存"""
        print(f"\n{'='*60}")
        print(f"显存测试 - {attention_type.upper()} 注意力")
        print(f"{'='*60}")

        model = TransformerEncoder(self.config, attention_type).to(self.device)

        # 模型显存分解
        print(f"\n模型显存分解:")
        model_mem = self.get_model_memory(model)
        print(f"  参数显存: {model_mem['params_mb']:.2f} MB")
        print(f"  缓冲区显存: {model_mem['buffers_mb']:.2f} MB")
        print(f"  总计: {model_mem['total_mb']:.2f} MB")

        results = {
            'attention_type': attention_type,
            'model_memory': model_mem,
            'batch_tests': {},
        }

        for batch_size in batch_sizes:
            print(f"\n批次大小: {batch_size}")

            try:
                # 前向传播显存
                forward_mem = self.measure_forward_memory(model, batch_size, seq_len)
                print(f"  前向传播峰值: {forward_mem['peak_mb']:.2f} MB")

                # 反向传播显存
                backward_mem = self.measure_backward_memory(model, batch_size, seq_len)
                print(f"  反向传播峰值: {backward_mem['peak_mb']:.2f} MB")

                # 完整训练显存
                full_mem = self.measure_full_training_memory(model, batch_size, seq_len)
                print(f"  完整训练峰值: {full_mem['peak_memory_mb']:.2f} MB ({full_mem['peak_memory_gb']:.2f} GB)")

                results['batch_tests'][f'bs{batch_size}'] = {
                    'forward': forward_mem,
                    'backward': backward_mem,
                    'full_training': full_mem,
                }

            except RuntimeError as e:
                print(f"  错误: 显存不足 - {e}")
                results['batch_tests'][f'bs{batch_size}'] = {'error': 'OOM'}

        del model
        torch.cuda.empty_cache()

        return results

    def run_benchmark(self, attention_type: str, batch_sizes: List[int], seq_len: int) -> Dict:
        """运行完整显存测试"""
        print(f"\n设备: {self.device}")
        print(f"模型配置: d_model={self.config.d_model}, n_heads={self.config.n_heads}, n_layers={self.config.n_layers}")

        results = self.test_batch_sizes(attention_type, seq_len, batch_sizes)

        return results


def main():
    parser = argparse.ArgumentParser(description='显存基准测试')
    parser.add_argument('--attention', type=str, default='orthogonal',
                        choices=['standard', 'orthogonal', 'synthesizer'],
                        help='注意力机制类型')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='批次大小列表')
    parser.add_argument('--seq_len', type=int, default=512, help='序列长度')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='层数')
    parser.add_argument('--output', type=str, default='memory_results.json', help='输出文件')

    args = parser.parse_args()

    config = MemoryConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
    )

    tester = MemoryTester(config)
    results = tester.run_benchmark(args.attention, args.batch_sizes, args.seq_len)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
