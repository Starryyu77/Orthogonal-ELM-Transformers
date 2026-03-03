"""
OELM-FFN: 扩展OELM到FFN层

在原有OELM基础上，将FFN的升维和降维矩阵也替换为冻结的正交矩阵。

支持的配置模式:
1. baseline: 全部可训练
2. oelm_qk: 只冻结Q/K
3. oelm_qk_ffn: 冻结Q/K和FFN
4. oelm_ffn_only: 只冻结FFN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# 导入正交FFN模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from orthogonal_ffn import FrozenOrthogonalLinear, OrthogonalFFN


class HeadWiseOrthogonalLinear(nn.Module):
    """
    分头正交初始化的Linear层（复用原实现）。
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        freeze: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.freeze = freeze
        self.init_method = init_method

        if init_method == 'orthogonal':
            weight = self._init_head_wise_orthogonal()
        elif init_method == 'normal':
            weight = torch.randn(d_model, d_model) * 0.02
        else:
            raise ValueError(f"未知init_method: {init_method}")

        if freeze:
            self.register_buffer('weight', weight)
        else:
            self.weight = nn.Parameter(weight.clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('bias', None)

    def _init_head_wise_orthogonal(self) -> torch.Tensor:
        """分头正交初始化。"""
        head_weights = []

        for _ in range(self.num_heads):
            A = torch.randn(self.d_model, self.head_dim)
            Q, R = torch.linalg.qr(A, mode='reduced')
            W = Q.T
            signs = torch.sign(torch.diag(R))
            W = W * signs.unsqueeze(1)
            head_weights.append(W)

        stacked = torch.stack(head_weights, dim=0)
        weight = stacked.view(self.d_model, self.d_model).contiguous()
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class HeadWiseOrthogonalMultiHeadAttention(nn.Module):
    """分头正交Multi-Head Attention（支持可配置冻结）。"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.head_dim = self.d_k
        self.freeze_qk = freeze_qk
        self.init_method = init_method

        # Q和K: 分头正交初始化
        self.W_q = HeadWiseOrthogonalLinear(
            d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method
        )
        self.W_k = HeadWiseOrthogonalLinear(
            d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method
        )

        # V: 正常初始化，可训练
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output: 可训练
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)

        return out


class OELMFFNTransformerLayer(nn.Module):
    """
    支持正交FFN的Transformer层。

    Args:
        freeze_qk: 是否冻结Q/K
        freeze_ffn: 是否冻结FFN
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        freeze_ffn: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()

        self.self_attn = HeadWiseOrthogonalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            freeze_qk=freeze_qk,
            init_method=init_method
        )

        # 使用正交FFN
        self.ffn = OrthogonalFFN(
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            freeze_ffn=freeze_ffn,
            init_method=init_method,
            activation='gelu'
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask, is_causal=False)
        x = x + self.dropout(attn_out)

        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class OELMFFNForSequenceClassification(nn.Module):
    """
    OELM-FFN用于序列分类。

    支持四种配置模式:
    - baseline: 全部可训练
    - oelm_qk: 只冻结Q/K
    - oelm_qk_ffn: 冻结Q/K和FFN
    - oelm_ffn_only: 只冻结FFN

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
        freeze_qk: 是否冻结Q/K
        freeze_ffn: 是否冻结FFN
        init_method: 'orthogonal'或'normal'
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
        pad_token_id: int = 0,
        freeze_qk: bool = True,
        freeze_ffn: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.pad_token_id = pad_token_id
        self.freeze_qk = freeze_qk
        self.freeze_ffn = freeze_ffn
        self.init_method = init_method

        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers
        self.layers = nn.ModuleList([
            OELMFFNTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                freeze_qk=freeze_qk,
                freeze_ffn=freeze_ffn,
                init_method=init_method
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # 分类头
        self.classifier = nn.Linear(d_model, num_classes)

        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        self.apply(self._init_weights)
        self._print_model_info()

    def _init_weights(self, module):
        """初始化权重（正交层除外）。"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """打印模型参数统计。"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffer_params = sum(b.numel() for b in self.buffers())
        all_params = trainable_params + buffer_params

        frozen_pct = 100 * buffer_params / all_params if all_params > 0 else 0
        trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

        # 确定模式名称
        if self.freeze_qk and self.freeze_ffn:
            mode = "oelm_qk_ffn"
        elif self.freeze_qk:
            mode = "oelm_qk"
        elif self.freeze_ffn:
            mode = "oelm_ffn_only"
        else:
            mode = "baseline"

        print(f"OELMFFNForSequenceClassification ({mode}):")
        print(f"  总参数量: {all_params:,}")
        print(f"  可训练参数量: {trainable_params:,} ({trainable_pct:.1f}%)")
        print(f"  冻结参数量: {buffer_params:,} ({frozen_pct:.1f}%)")
        print(f"  分类类别数: {self.num_classes}")
        print(f"  Q/K冻结: {self.freeze_qk}")
        print(f"  FFN冻结: {self.freeze_ffn}")
        print(f"  初始化: {self.init_method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=attention_mask)

        x = self.norm(x)

        # 池化: 取最后一个有效token
        if attention_mask is not None:
            last_valid_indices = attention_mask.sum(dim=1).long() - 1
            pooled = x[torch.arange(batch_size), last_valid_indices]
        else:
            pooled = x[:, -1]

        logits = self.classifier(pooled)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss

        return logits

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """获取最后一层hidden states。"""
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)

        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, mask=None)

        x = self.norm(x)
        return x


def create_oelm_ffn_classifier(
    num_classes: int,
    vocab_size: int = 50257,
    d_model: int = 512,
    num_layers: int = 6,
    num_heads: int = 8,
    freeze_qk: bool = True,
    freeze_ffn: bool = True,
    init_method: str = 'orthogonal',
    **kwargs
) -> OELMFFNForSequenceClassification:
    """
    创建OELM-FFN分类模型。

    Args:
        freeze_qk: 是否冻结Q/K
        freeze_ffn: 是否冻结FFN
    """
    return OELMFFNForSequenceClassification(
        num_classes=num_classes,
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        freeze_qk=freeze_qk,
        freeze_ffn=freeze_ffn,
        init_method=init_method,
        **kwargs
    )


# 便捷的工厂函数
def create_baseline_classifier(**kwargs):
    """创建Baseline分类器（全部可训练）。"""
    return create_oelm_ffn_classifier(
        freeze_qk=False, freeze_ffn=False, **kwargs
    )


def create_oelm_qk_classifier(**kwargs):
    """创建OELM-QK分类器（只冻结Q/K）。"""
    return create_oelm_ffn_classifier(
        freeze_qk=True, freeze_ffn=False, **kwargs
    )


def create_oelm_qk_ffn_classifier(**kwargs):
    """创建OELM-QK-FFN分类器（冻结Q/K和FFN）。"""
    return create_oelm_ffn_classifier(
        freeze_qk=True, freeze_ffn=True, **kwargs
    )


def create_oelm_ffn_only_classifier(**kwargs):
    """创建OELM-FFN-only分类器（只冻结FFN）。"""
    return create_oelm_ffn_classifier(
        freeze_qk=False, freeze_ffn=True, **kwargs
    )


if __name__ == "__main__":
    print("测试OELM-FFN分类模型...\n")

    # Test 1: OELM-QK-FFN
    print("Test 1: OELM-QK-FFN (冻结Q/K和FFN)")
    model = create_oelm_qk_ffn_classifier(
        num_classes=2,
        vocab_size=1000,
        init_method='orthogonal'
    )
    input_ids = torch.randint(0, 1000, (4, 32))
    attention_mask = torch.ones(4, 32)
    attention_mask[:, 25:] = 0

    logits = model(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 2)
    print(f"  Logits shape: {logits.shape}")
    print("  ✓ 通过\n")

    # Test 2: OELM-FFN-only
    print("Test 2: OELM-FFN-only (只冻结FFN)")
    model_ffn_only = create_oelm_ffn_only_classifier(
        num_classes=4,
        vocab_size=1000,
        init_method='orthogonal'
    )
    logits = model_ffn_only(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 4)
    print(f"  Logits shape: {logits.shape}")
    print("  ✓ 通过\n")

    # Test 3: Baseline
    print("Test 3: Baseline (全部可训练)")
    model_baseline = create_baseline_classifier(
        num_classes=2,
        vocab_size=1000
    )
    logits = model_baseline(input_ids, attention_mask=attention_mask)
    assert logits.shape == (4, 2)
    print(f"  Logits shape: {logits.shape}")
    print("  ✓ 通过\n")

    # Test 4: 带loss的前向传播
    print("Test 4: 带loss的前向传播")
    labels = torch.randint(0, 2, (4,))
    loss = model(input_ids, attention_mask=attention_mask, labels=labels)
    assert loss.ndim == 0
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ 通过\n")

    print("所有测试通过！✓")
