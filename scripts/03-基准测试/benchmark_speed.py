#!/usr/bin/env python3
"""
测速脚本 (benchmark_speed.py)
=============================

用于测试正交随机注意力的训练和推理速度。

功能:
1. 训练速度测试
2. 推理速度测试  
3. 吞吐量计算

用法:
    python benchmark_speed.py --attention orthogonal --batch_size 8 --seq_len 512
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import argparse
from typing import Dict, Optional
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class SpeedConfig:
    """速度测试配置"""
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    max_seq_len: int = 512
    vocab_size: int = 10000
    dropout: float = 0.1
    num_random_features: int = 256
    num_iterations: int = 100
    warmup_iterations: int = 10
    device: str = 'cuda'


# =============================================================================
# 正交随机注意力实现
# =============================================================================

class OrthogonalRandomFeatures(nn.Module):
    """正交随机特征"""
    def __init__(self, d_model: int, num_features: int):
        super().__init__()
        self.d_model = d_model
        self.num_features = num_features

        # 初始化正交随机投影矩阵
        W = torch.randn(d_model, num_features)
        Q, _ = torch.linalg.qr(W)
        self.register_buffer('omega', Q[:, :num_features])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = torch.matmul(x, self.omega.to(x.device))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class OrthogonalRandomAttention(nn.Module):
    """正交随机注意力机制"""
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

        # 应用正交随机特征
        Q_features = self.ortho_features(Q.transpose(1, 2).reshape(-1, seq_len, self.d_head))
        K_features = self.ortho_features(K.transpose(1, 2).reshape(-1, seq_len, self.d_head))

        Q_features = Q_features.view(batch_size, self.n_heads, seq_len, self.num_features)
        K_features = K_features.view(batch_size, self.n_heads, seq_len, self.num_features)

        # 近似注意力计算
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
    def __init__(self, config: SpeedConfig, attention_type: str = 'standard'):
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
# 速度测试器
# =============================================================================

class SpeedTester:
    """速度测试器"""

    def __init__(self, config: SpeedConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    def warmup(self, model: nn.Module, batch_size: int, seq_len: int):
        """GPU预热"""
        model.eval()
        dummy_input = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(dummy_input)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def test_training_speed(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测试训练速度"""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss()

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)
        labels = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        # 预热
        for _ in range(self.config.warmup_iterations):
            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, self.config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 正式测试
        times = []
        for _ in range(self.config.num_iterations):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            optimizer.zero_grad()
            output = model(input_ids)
            loss = criterion(output.view(-1, self.config.vocab_size), labels.view(-1))
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'samples_per_sec': batch_size / avg_time,
            'tokens_per_sec': batch_size * seq_len / avg_time,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
        }

    def test_inference_speed(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测试推理速度"""
        model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 正式测试
        times = []
        with torch.no_grad():
            for _ in range(self.config.num_iterations):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                _ = model(input_ids)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                times.append(time.perf_counter() - start)

        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5

        return {
            'avg_time_ms': avg_time * 1000,
            'std_time_ms': std_time * 1000,
            'samples_per_sec': batch_size / avg_time,
            'tokens_per_sec': batch_size * seq_len / avg_time,
            'min_time_ms': min(times) * 1000,
            'max_time_ms': max(times) * 1000,
        }

    def run_benchmark(self, attention_type: str, batch_size: int, seq_len: int) -> Dict:
        """运行完整速度测试"""
        print(f"\n{'='*60}")
        print(f"速度测试 - {attention_type.upper()} 注意力")
        print(f"{'='*60}")

        model = TransformerEncoder(self.config, attention_type).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"测试配置: batch_size={batch_size}, seq_len={seq_len}")

        # 预热
        print("预热中...")
        self.warmup(model, batch_size, seq_len)

        # 训练速度测试
        print(f"\n训练速度测试 ({self.config.num_iterations} 次迭代)...")
        train_result = self.test_training_speed(model, batch_size, seq_len)
        print(f"  平均时间: {train_result['avg_time_ms']:.2f} ± {train_result['std_time_ms']:.2f} ms")
        print(f"  吞吐量: {train_result['samples_per_sec']:.2f} samples/s, {train_result['tokens_per_sec']:.0f} tokens/s")

        # 推理速度测试
        print(f"\n推理速度测试 ({self.config.num_iterations} 次迭代)...")
        infer_result = self.test_inference_speed(model, batch_size, seq_len)
        print(f"  平均时间: {infer_result['avg_time_ms']:.2f} ± {infer_result['std_time_ms']:.2f} ms")
        print(f"  吞吐量: {infer_result['samples_per_sec']:.2f} samples/s, {infer_result['tokens_per_sec']:.0f} tokens/s")

        del model
        torch.cuda.empty_cache()

        return {
            'attention_type': attention_type,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'training': train_result,
            'inference': infer_result,
        }


def main():
    parser = argparse.ArgumentParser(description='速度基准测试')
    parser.add_argument('--attention', type=str, default='orthogonal',
                        choices=['standard', 'orthogonal', 'synthesizer'],
                        help='注意力机制类型')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--seq_len', type=int, default=512, help='序列长度')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='层数')
    parser.add_argument('--iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--output', type=str, default='speed_results.json', help='输出文件')

    args = parser.parse_args()

    config = SpeedConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        num_iterations=args.iterations,
    )

    print(f"\n设备: {config.device}")
    print(f"模型配置: d_model={config.d_model}, n_heads={config.n_heads}, n_layers={config.n_layers}")

    tester = SpeedTester(config)
    results = tester.run_benchmark(args.attention, args.batch_size, args.seq_len)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n结果已保存: {args.output}")


if __name__ == '__main__':
    main()
