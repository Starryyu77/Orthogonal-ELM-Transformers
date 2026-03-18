"""
OELM for Language Modeling (Pre-training)

基于phase2/phase4的分类模型改造为预训练版本。
支持:
1. Baseline: 标准GPT预训练
2. OELM-QK: 冻结Q/K
3. OELM-QK-FFN: 冻结Q/K+FFN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class OELMConfig:
    """OELM预训练模型配置"""

    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1

    # OELM特定配置
    freeze_qk: bool = True
    freeze_ffn: bool = False
    init_method: str = "orthogonal"  # 'orthogonal' or 'normal'

    # 模型规模预设
    @classmethod
    def small(cls):
        """GPT-Small: 117M params"""
        return cls(
            d_model=768, num_layers=12, num_heads=12, d_ff=3072, vocab_size=50257
        )

    @classmethod
    def medium(cls):
        """GPT-Medium: 355M params"""
        return cls(
            d_model=1024, num_layers=24, num_heads=16, d_ff=4096, vocab_size=50257
        )

    def get_param_count(self):
        """估算参数量"""
        # Embeddings
        embed_params = self.vocab_size * self.d_model * 2  # token + position

        # Per layer
        # - Attention: Q, K, V, O (each d_model x d_model)
        # - FFN: Up (d_model x d_ff), Down (d_ff x d_model)
        # - LayerNorm: 2 * d_model per norm
        attn_params = 4 * self.d_model * self.d_model
        ffn_params = 2 * self.d_model * self.d_ff
        ln_params = 4 * self.d_model
        layer_params = attn_params + ffn_params + ln_params

        total_params = embed_params + self.num_layers * layer_params
        return total_params


class FrozenOrthogonalLinear(nn.Module):
    """
    冻结的正交线性层（用于FFN）
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        freeze: bool = True,
    ):
        super().__init__()

        # 正交初始化
        weight = torch.empty(out_features, in_features)
        nn.init.orthogonal_(weight)

        if freeze:
            self.register_buffer("weight", weight)
        else:
            self.weight = nn.Parameter(weight.clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class HeadWiseOrthogonalLinear(nn.Module):
    """
    分头正交初始化的Linear层
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        bias: bool = False,
        freeze: bool = True,
        init_method: str = "orthogonal",
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.freeze = freeze
        self.init_method = init_method

        # 根据初始化方法初始化权重
        if init_method == "orthogonal":
            weight = self._init_head_wise_orthogonal()
        elif init_method == "normal":
            weight = torch.randn(d_model, d_model) * 0.02
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # 注册为buffer（冻结）或parameter（可训练）
        if freeze:
            self.register_buffer("weight", weight)
        else:
            self.weight = nn.Parameter(weight.clone())

        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter("bias", None)

    def _init_head_wise_orthogonal(self) -> torch.Tensor:
        """分头正交初始化"""
        head_weights = []

        for _ in range(self.num_heads):
            A = torch.randn(self.d_model, self.head_dim)
            Q, R = torch.linalg.qr(A, mode="reduced")
            W = Q.T
            signs = torch.sign(torch.diag(R))
            W = W * signs.unsqueeze(1)
            head_weights.append(W)

        stacked = torch.stack(head_weights, dim=0)
        weight = stacked.view(self.d_model, self.d_model).contiguous()
        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class OELMMultiHeadAttention(nn.Module):
    """OELM Multi-Head Attention with causal masking"""

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        init_method: str = "orthogonal",
    ):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q和K: 分头正交初始化
        self.W_q = HeadWiseOrthogonalLinear(
            d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method
        )
        self.W_k = HeadWiseOrthogonalLinear(
            d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method
        )

        # V和O: 可训练
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Causal mask for autoregressive language modeling
        if mask is None:
            # Create causal mask
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            scores = scores.masked_fill(
                causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
            )
        else:
            scores = scores.masked_fill(
                mask.unsqueeze(1).unsqueeze(1) == 0, float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)

        return out


class OELMFFN(nn.Module):
    """FFN with optional frozen orthogonal weights"""

    def __init__(
        self, d_model: int, d_ff: int, dropout: float = 0.1, freeze_ffn: bool = False
    ):
        super().__init__()

        if freeze_ffn:
            # 使用冻结的正交矩阵
            self.up_proj = FrozenOrthogonalLinear(d_model, d_ff, bias=True, freeze=True)
            self.down_proj = FrozenOrthogonalLinear(
                d_ff, d_model, bias=True, freeze=True
            )
        else:
            # 标准可训练FFN
            self.up_proj = nn.Linear(d_model, d_ff)
            self.down_proj = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class OELMTransformerBlock(nn.Module):
    """Transformer block with pre-norm"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        freeze_ffn: bool = False,
        init_method: str = "orthogonal",
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(d_model)
        self.attn = OELMMultiHeadAttention(
            d_model, num_heads, dropout, freeze_qk, init_method
        )

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = OELMFFN(d_model, d_ff, dropout, freeze_ffn)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention
        x = x + self.dropout(self.attn(self.ln1(x), mask))
        # Pre-norm FFN
        x = x + self.ffn(self.ln2(x))
        return x


class OELMForLanguageModeling(nn.Module):
    """
    OELM for Language Modeling (Pre-training)

    支持三种模式:
    1. Baseline: freeze_qk=False, freeze_ffn=False
    2. OELM-QK: freeze_qk=True, freeze_ffn=False
    3. OELM-QK-FFN: freeze_qk=True, freeze_ffn=True
    """

    def __init__(self, config: OELMConfig):
        super().__init__()
        self.config = config

        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [
                OELMTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    config.dropout,
                    config.freeze_qk,
                    config.freeze_ffn,
                    config.init_method,
                )
                for _ in range(config.num_layers)
            ]
        )

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)

        # Language modeling head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)
        self._print_model_info()

    def _init_weights(self, module):
        """Initialize weights (excluding frozen orthogonal layers)"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def _print_model_info(self):
        """Print model parameter statistics"""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffer_params = sum(b.numel() for b in self.buffers())
        total_params = trainable_params + buffer_params

        print(f"\nOELMForLanguageModeling:")
        print(
            f"  Model size: d_model={self.config.d_model}, layers={self.config.num_layers}, heads={self.config.num_heads}"
        )
        print(f"  Total parameters: {total_params:,}")
        print(
            f"  Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)"
        )
        print(
            f"  Frozen parameters: {buffer_params:,} ({100 * buffer_params / total_params:.1f}%)"
        )
        print(
            f"  Config: freeze_qk={self.config.freeze_qk}, freeze_ffn={self.config.freeze_ffn}, init={self.config.init_method}"
        )

    def forward(
        self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass

        Args:
            input_ids: Token ids [batch_size, seq_len]
            labels: Target labels for language modeling [batch_size, seq_len]

        Returns:
            logits: Output logits [batch_size, seq_len, vocab_size]
            loss: Language modeling loss if labels provided
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position indices
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device).unsqueeze(
            0
        )

        # Embeddings
        token_embeds = self.token_embedding(input_ids)
        pos_embeds = self.position_embedding(positions)
        x = self.dropout(token_embeds + pos_embeds)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift logits and labels for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> torch.Tensor:
        """Simple text generation"""
        for _ in range(max_new_tokens):
            # Get logits for last token
            logits, _ = self.forward(input_ids)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


def create_model(
    model_size: str = "small", method: str = "baseline", vocab_size: int = 50257
) -> OELMForLanguageModeling:
    """
    Factory function to create OELM models

    Args:
        model_size: 'small' (117M) or 'medium' (355M)
        method: 'baseline', 'oelm_qk', or 'oelm_qk_ffn'
        vocab_size: Vocabulary size

    Returns:
        OELMForLanguageModeling model
    """
    # Get base config
    if model_size == "small":
        config = OELMConfig.small()
    elif model_size == "medium":
        config = OELMConfig.medium()
    else:
        raise ValueError(f"Unknown model_size: {model_size}")

    config.vocab_size = vocab_size

    # Configure method
    if method == "baseline":
        config.freeze_qk = False
        config.freeze_ffn = False
        config.init_method = "normal"
    elif method == "oelm_qk":
        config.freeze_qk = True
        config.freeze_ffn = False
        config.init_method = "orthogonal"
    elif method == "oelm_qk_ffn":
        config.freeze_qk = True
        config.freeze_ffn = True
        config.init_method = "orthogonal"
    else:
        raise ValueError(f"Unknown method: {method}")

    return OELMForLanguageModeling(config)


if __name__ == "__main__":
    # Test model creation
    print("Testing OELM pretrain model...")

    for method in ["baseline", "oelm_qk", "oelm_qk_ffn"]:
        print(f"\n{'=' * 60}")
        print(f"Method: {method}")
        print("=" * 60)

        model = create_model("small", method)

        # Test forward pass
        input_ids = torch.randint(0, 50257, (2, 128))
        labels = input_ids.clone()

        logits, loss = model(input_ids, labels)
        print(f"Input shape: {input_ids.shape}")
        print(f"Logits shape: {logits.shape}")
        print(f"Loss: {loss.item():.4f}")
