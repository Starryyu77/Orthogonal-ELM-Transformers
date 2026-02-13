"""
GPT for Sequence Classification (Baseline)

将GPT改造为分类模型，用于验证OELM在GPT+分类任务上的效果。
使用双向attention（非因果），适用于分类任务。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    标准Multi-Head Attention，支持双向attention（分类任务）。
    所有参数可训练（Baseline）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q, K, V projections: 全部可训练
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: 可训练
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False  # 分类任务使用双向attention
    ) -> torch.Tensor:
        """前向传播。"""
        batch_size, seq_len, _ = x.shape

        # 计算Q, K, V
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # reshape为多head
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 计算attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 分类任务不使用causal mask (is_causal=False)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))

        # softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用到values
        out = torch.matmul(attn_weights, V)

        # 合并heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)

        return out


class GPTTransformerLayer(nn.Module):
    """标准GPT Transformer层（双向attention版本）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """前向传播。"""
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask, is_causal=False)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class GPTForSequenceClassification(nn.Module):
    """
    GPT用于序列分类任务。

    改造点:
    1. 移除语言建模头 (lm_head)
    2. 添加分类头: Linear(d_model, num_classes)
    3. 使用双向attention (非因果)
    4. 取[CLS] token或最后一个有效token的hidden state

    Args:
        num_classes: 分类类别数
        vocab_size: 词表大小
        d_model: 模型维度
        num_layers: Transformer层数
        num_heads: Attention head数
        d_ff: Feed-forward维度
        max_seq_len: 最大序列长度
        dropout: Dropout概率
        pad_token_id: padding token ID
    """

    def __init__(
        self,
        num_classes: int,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        super().__init__()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers (双向attention)
        self.layers = nn.ModuleList([
            GPTTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 分类头 (替换原来的lm_head)
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        self.apply(self._init_weights)
        self._print_model_info()

    def _init_weights(self, module):
        """初始化权重。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """打印模型参数统计。"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"GPTForSequenceClassification (Baseline):")
        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数量: {trainable_params:,} (100.0%)")
        print(f"  分类类别数: {self.num_classes}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播。

        Args:
            input_ids: 输入token IDs (batch_size, seq_len)
            attention_mask: 注意力mask (batch_size, seq_len)
            labels: 分类标签 (batch_size,)

        Returns:
            如果labels为None，返回logits
            如果labels不为None，返回loss
        """
        batch_size, seq_len = input_ids.shape

        # 创建position IDs
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        # Embedding
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transformer layers (双向attention)
        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)

        # 池化: 取最后一个有效token的hidden state
        if attention_mask is not None:
            # 找到每个序列最后一个非padding位置
            # attention_mask: 1表示有效token，0表示padding
            last_valid_indices = attention_mask.sum(dim=1).long() - 1  # (batch_size,)
            # 提取每个序列最后一个有效token的表示
            pooled = x[torch.arange(batch_size), last_valid_indices]
        else:
            # 没有mask时取最后一个token
            pooled = x[:, -1]

        # 分类
        logits = self.classifier(pooled)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取最后一层hidden states（用于分析）。"""
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=None)

        x = self.norm(x)
        return x


def create_gpt_classifier(
    num_classes: int,
    vocab_size: int = 50257,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    **kwargs
) -> GPTForSequenceClassification:
    """创建GPT分类模型。"""
    return GPTForSequenceClassification(
        num_classes=num_classes,
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        **kwargs
    )


if __name__ == "__main__":
    print("测试GPT分类模型...\n")

    # Test 1: 基础前向传播
    print("Test 1: 基础前向传播")
    model = create_gpt_classifier(num_classes=2, vocab_size=1000)
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32)
    attention_mask[:, 25:] = 0  # 模拟padding

    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 2), f"期望(4, 2)，得到{logits.shape}"
    print(f"  Logits shape: {logits.shape}")
    print("  ✓ 通过\n")

    # Test 2: 带loss的前向传播
    print("Test 2: 带loss的前向传播")
    labels = torch.randint(0, 2, (4,))
    loss = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert loss.ndim == 0
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ 通过\n")

    # Test 3: 不同类别数
    print("Test 3: 不同类别数")
    model_4cls = create_gpt_classifier(num_classes=4, vocab_size=1000)
    logits_4cls = model_4cls(input_ids, attention_mask=attention_mask)
    assert logits_4cls.shape == (4, 4)
    print(f"  4分类 logits shape: {logits_4cls.shape}")
    print("  ✓ 通过\n")

    print("所有测试通过！✓")
