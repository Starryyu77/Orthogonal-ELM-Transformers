#!/usr/bin/env python3
"""
对比测试脚本 (benchmark_compare.py)
====================================

用于对比不同注意力机制的性能。

功能:
1. 与标准Transformer对比
2. 与Synthesizer对比
3. 生成对比报告

用法:
    python benchmark_compare.py --output_dir ./results
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import os
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class CompareConfig:
    """对比测试配置"""
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
    batch_sizes: List[int] = None
    seq_lengths: List[int] = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16]
        if self.seq_lengths is None:
            self.seq_lengths = [128, 256, 512]


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
    def __init__(self, config: CompareConfig, attention_type: str = 'standard'):
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
# 对比测试器
# =============================================================================

class ComparisonTester:
    """对比测试器"""

    def __init__(self, config: CompareConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')

    def measure_speed(self, model: nn.Module, batch_size: int, seq_len: int, mode: str = 'inference') -> Dict:
        """测量速度"""
        model.eval() if mode == 'inference' else model.train()

        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        # 预热
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # 测试
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

        return {
            'avg_time_ms': avg_time * 1000,
            'tokens_per_sec': batch_size * seq_len / avg_time,
        }

    def measure_memory(self, model: nn.Module, batch_size: int, seq_len: int) -> Dict:
        """测量显存"""
        if not torch.cuda.is_available():
            return {'peak_memory_mb': 0}

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model.eval()
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len)).to(self.device)

        with torch.no_grad():
            _ = model(input_ids)

        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

        return {'peak_memory_mb': peak_mem}

    def compare_single_config(self, batch_size: int, seq_len: int) -> Dict:
        """对比单一配置"""
        print(f"\n测试配置: batch_size={batch_size}, seq_len={seq_len}")

        results = {}

        for attn_type in ['standard', 'orthogonal', 'synthesizer']:
            model = TransformerEncoder(self.config, attn_type).to(self.device)
            total_params = sum(p.numel() for p in model.parameters())

            # 速度测试
            speed = self.measure_speed(model, batch_size, seq_len)

            # 显存测试
            memory = self.measure_memory(model, batch_size, seq_len)

            results[attn_type] = {
                'params': total_params,
                'speed': speed,
                'memory': memory,
            }

            print(f"  {attn_type:12} - 速度: {speed['avg_time_ms']:>7.2f} ms, "
                  f"吞吐量: {speed['tokens_per_sec']:>8.0f} tokens/s, "
                  f"显存: {memory['peak_memory_mb']:>8.2f} MB")

            del model
            torch.cuda.empty_cache()

        return results

    def run_comparison(self) -> Dict:
        """运行完整对比"""
        print(f"\n{'='*70}")
        print("对比测试 - 正交随机注意力 vs 标准Transformer vs Synthesizer")
        print(f"{'='*70}")

        print(f"\n设备: {self.device}")
        print(f"模型配置: d_model={self.config.d_model}, n_heads={self.config.n_heads}, n_layers={self.config.n_layers}")

        all_results = {}

        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.seq_lengths:
                if batch_size * seq_len > 65536:
                    continue

                key = f"bs{batch_size}_sl{seq_len}"
                all_results[key] = self.compare_single_config(batch_size, seq_len)

        return all_results

    def generate_markdown_report(self, results: Dict, output_path: str):
        """生成Markdown格式报告"""
        lines = []

        lines.append("# 正交随机注意力基准测试对比报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        lines.append("\n## 测试配置")
        lines.append(f"- d_model: {self.config.d_model}")
        lines.append(f"- n_heads: {self.config.n_heads}")
        lines.append(f"- n_layers: {self.config.n_layers}")
        lines.append(f"- max_seq_len: {self.config.max_seq_len}")
        lines.append(f"- num_iterations: {self.config.num_iterations}")

        lines.append("\n## 速度对比")
        lines.append("\n### 推理速度 (ms/batch)")
        lines.append("| Config | Standard | Orthogonal | Synthesizer |")
        lines.append("|--------|----------|------------|-------------|")

        for key, data in results.items():
            std_time = data['standard']['speed']['avg_time_ms']
            ortho_time = data['orthogonal']['speed']['avg_time_ms']
            synth_time = data['synthesizer']['speed']['avg_time_ms']
            lines.append(f"| {key} | {std_time:.2f} | {ortho_time:.2f} | {synth_time:.2f} |")

        lines.append("\n### 吞吐量 (tokens/sec)")
        lines.append("| Config | Standard | Orthogonal | Synthesizer |")
        lines.append("|--------|----------|------------|-------------|")

        for key, data in results.items():
            std_tps = data['standard']['speed']['tokens_per_sec']
            ortho_tps = data['orthogonal']['speed']['tokens_per_sec']
            synth_tps = data['synthesizer']['speed']['tokens_per_sec']
            lines.append(f"| {key} | {std_tps:.0f} | {ortho_tps:.0f} | {synth_tps:.0f} |")

        lines.append("\n## 显存对比")
        lines.append("| Config | Standard | Orthogonal | Synthesizer |")
        lines.append("|--------|----------|------------|-------------|")

        for key, data in results.items():
            std_mem = data['standard']['memory']['peak_memory_mb']
            ortho_mem = data['orthogonal']['memory']['peak_memory_mb']
            synth_mem = data['synthesizer']['memory']['peak_memory_mb']
            lines.append(f"| {key} | {std_mem:.2f} | {ortho_mem:.2f} | {synth_mem:.2f} |")

        lines.append("\n## 参数量对比")
        lines.append("| Attention Type | Parameters |")
        lines.append("|----------------|------------|")

        first_key = list(results.keys())[0]
        for attn_type in ['standard', 'orthogonal', 'synthesizer']:
            params = results[first_key][attn_type]['params']
            lines.append(f"| {attn_type.capitalize()} | {params:,} ({params/1e6:.2f}M) |")

        lines.append("\n## 性能提升分析")

        for key, data in results.items():
            std_time = data['standard']['speed']['avg_time_ms']
            ortho_time = data['orthogonal']['speed']['avg_time_ms']
            speedup = (std_time - ortho_time) / std_time * 100

            lines.append(f"\n### {key}")
            lines.append(f"- 正交随机注意力相比标准Transformer速度提升: {speedup:.1f}%")

            if speedup > 0:
                lines.append(f"  - 正交随机注意力更快")
            else:
                lines.append(f"  - 标准Transformer更快")

        report = "\n".join(lines)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\n报告已保存: {output_path}")
        return report

    def generate_json_report(self, results: Dict, output_path: str):
        """生成JSON格式报告"""
        report = {
            'config': asdict(self.config),
            'timestamp': datetime.now().isoformat(),
            'results': results,
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"JSON报告已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='对比基准测试')
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--n_layers', type=int, default=6, help='层数')
    parser.add_argument('--seq_len', type=int, default=512, help='序列长度')
    parser.add_argument('--batch_sizes', type=int, nargs='+', default=[1, 4, 8],
                        help='批次大小列表')
    parser.add_argument('--iterations', type=int, default=100, help='迭代次数')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = CompareConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.seq_len,
        num_iterations=args.iterations,
        batch_sizes=args.batch_sizes,
        seq_lengths=[args.seq_len],
    )

    tester = ComparisonTester(config)
    results = tester.run_comparison()

    # 生成报告
    md_path = os.path.join(args.output_dir, 'comparison_report.md')
    tester.generate_markdown_report(results, md_path)

    json_path = os.path.join(args.output_dir, 'comparison_results.json')
    tester.generate_json_report(results, json_path)


if __name__ == '__main__':
    main()
