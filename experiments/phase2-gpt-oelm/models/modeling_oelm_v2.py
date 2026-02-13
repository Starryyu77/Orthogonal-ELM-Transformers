"""
Orthogonal ELM Transformer v2 - Head-wise Orthogonal Initialization

关键修正：从全局正交改为分头正交初始化
- 之前：对整个 [d_model, d_model] 权重矩阵做 QR 分解
- 现在：将权重 reshape 为 [num_heads, head_dim, d_model]，对每个 head 单独做 QR

这个修正是基于 BERT OELM 实验的成功经验。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class HeadWiseOrthogonalLinear(nn.Module):
    """
    Linear layer with HEAD-WISE orthogonal initialization.

    关键创新：
    1. 将权重从 [d_model, d_model] reshape 为 [num_heads, head_dim, d_model]
    2. 对每个 head 的 [head_dim, d_model] 子矩阵独立进行 QR 分解
    3. 这样确保每个 head 内部是正交的，但不同 heads 之间可以独立变化

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        bias: Whether to include bias
        freeze: Whether to freeze the weight
        init_method: 'orthogonal' (for OELM-Freeze) or 'normal' (for OELM-Random ablation)
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
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.freeze = freeze
        self.init_method = init_method

        # Initialize weight based on init_method
        if init_method == 'orthogonal':
            weight = self._init_head_wise_orthogonal()
        elif init_method == 'normal':
            # Standard Gaussian initialization (for OELM-Random ablation)
            weight = torch.randn(d_model, d_model) * 0.02
        else:
            raise ValueError(f"Unknown init_method: {init_method}. Choose 'orthogonal' or 'normal'.")

        # Register as buffer (frozen) or parameter (trainable)
        if freeze:
            self.register_buffer('weight', weight)
        else:
            self.weight = nn.Parameter(weight.clone())

        # Bias remains trainable
        if bias:
            self.bias = nn.Parameter(torch.zeros(d_model))
        else:
            self.register_parameter('bias', None)

    def _init_head_wise_orthogonal(self) -> torch.Tensor:
        """
        Initialize weight with HEAD-WISE orthogonality.

        Returns:
            Weight tensor of shape [d_model, d_model] where each head's
            [head_dim, d_model] sub-matrix is row-orthogonal (W @ W.T = I)
        """
        # Create weight for each head: [num_heads, head_dim, d_model]
        head_weights = []

        for _ in range(self.num_heads):
            # Each head: [head_dim, d_model]
            # For row-orthogonal: we want W @ W.T = I_head_dim
            # Use QR on transpose: A.T = Q @ R, then Q.T gives row-orthogonal
            A = torch.randn(self.d_model, self.head_dim)  # [d_model, head_dim]
            Q, R = torch.linalg.qr(A, mode='reduced')  # Q: [d_model, head_dim], R: [head_dim, head_dim]
            # Q has orthonormal columns: Q.T @ Q = I
            # We want row-orthogonal: W @ W.T = I, so W = Q.T
            W = Q.T  # [head_dim, d_model]
            # Adjust signs for determinism
            signs = torch.sign(torch.diag(R))
            W = W * signs.unsqueeze(1)  # broadcasting: [head_dim, 1] * [head_dim, d_model]
            head_weights.append(W)

        # Stack all heads: [num_heads, head_dim, d_model]
        stacked = torch.stack(head_weights, dim=0)

        # Reshape to [d_model, d_model] = [num_heads * head_dim, d_model]
        weight = stacked.view(self.d_model, self.d_model).contiguous()

        return weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, num_heads={self.num_heads}, freeze={self.freeze}'


class HeadWiseOrthogonalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with HEAD-WISE orthogonal random Q/K projections.

    关键修正：
    - 使用 HeadWiseOrthogonalLinear 替代全局 OrthogonalLinear
    - 每个 head 独立正交初始化，保持 head 内部几何结构

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        freeze_qk: Whether to freeze Q/K projection matrices
        init_method: 'orthogonal' (for OELM-Freeze) or 'normal' (for OELM-Random ablation)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.head_dim = self.d_k  # head_dim is same as d_k
        self.freeze_qk = freeze_qk
        self.init_method = init_method

        # Q and K projections: HEAD-WISE orthogonal initialization, frozen
        self.W_q = HeadWiseOrthogonalLinear(d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method)
        self.W_k = HeadWiseOrthogonalLinear(d_model, num_heads, bias=False, freeze=freeze_qk, init_method=init_method)

        # V projection: normal initialization, trainable
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection: trainable
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

        # Verify head-wise orthogonality (only for orthogonal init)
        if init_method == 'orthogonal':
            self._verify_head_wise_orthogonality()

    def _verify_head_wise_orthogonality(self):
        """Verify that each head maintains orthogonality."""
        with torch.no_grad():
            for name, W in [('Q', self.W_q.weight), ('K', self.W_k.weight)]:
                # Reshape to [num_heads, head_dim, d_model]
                W_heads = W.view(self.num_heads, self.head_dim, self.d_model)

                # Check orthogonality for each head
                max_error = 0
                for h in range(self.num_heads):
                    head_weight = W_heads[h]  # [head_dim, d_model]
                    # For row-orthogonal: W @ W^T should be close to I
                    I = head_weight @ head_weight.T
                    error = torch.norm(I - torch.eye(self.head_dim, device=W.device), p='fro')
                    max_error = max(max_error, error.item())

                if max_error > 1e-3:
                    print(f"Warning: Head-wise orthogonality error for {name}: {max_error:.6f}")
                else:
                    print(f"  {name}: Head-wise orthogonality verified (max error: {max_error:.6f})")

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of head-wise orthogonal multi-head attention.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking

        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V projections
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model) - trainable

        # Reshape for multi-head attention: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # Compute attention scores: (batch, num_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply causal mask for autoregressive modeling
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Apply custom mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Compute attention weights and apply to V
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Final linear projection
        out = self.W_o(out)

        return out


class HeadWiseOrthogonalTransformerLayer(nn.Module):
    """
    Transformer layer with Head-wise Orthogonal ELM Attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()

        # Head-wise orthogonal multi-head attention
        self.self_attn = HeadWiseOrthogonalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            freeze_qk=freeze_qk,
            init_method=init_method
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer normalization (pre-norm)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass."""
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask, is_causal=is_causal)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x


class HeadWiseOrthogonalELMTransformer(nn.Module):
    """
    Complete Transformer model with HEAD-WISE Orthogonal ELM Attention.

    这是 GPT OELM 的 v2 版本，使用分头正交初始化替代全局正交初始化。

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        freeze_qk: Whether to freeze Q/K projection matrices
        init_method: 'orthogonal' (for OELM-Freeze) or 'normal' (for OELM-Random ablation)
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        freeze_qk: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.freeze_qk = freeze_qk
        self.init_method = init_method

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers with head-wise orthogonal attention
        self.layers = nn.ModuleList([
            HeadWiseOrthogonalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                freeze_qk=freeze_qk,
                init_method=init_method
            )
            for _ in range(num_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.token_embedding.weight

        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len

        # Initialize
        self.apply(self._init_weights)

        # Print info
        self._print_model_info()

    def _init_weights(self, module):
        """Initialize weights (except orthogonal layers)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _print_model_info(self):
        """Print model parameter statistics."""
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        buffer_params = sum(b.numel() for b in self.buffers())
        all_params = trainable_params + buffer_params

        frozen_pct = 100 * buffer_params / all_params if all_params > 0 else 0
        trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

        print(f"\nHeadWiseOrthogonalELMTransformer (v2):")
        print(f"  Total parameters: {all_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.1f}%)")
        print(f"  Frozen parameters: {buffer_params:,} ({frozen_pct:.1f}%)")
        print(f"  Q/K frozen: {self.freeze_qk}")
        print(f"  Initialization: HEAD-WISE {self.init_method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> torch.Tensor:
        """Forward pass."""
        batch_size, seq_len = input_ids.shape

        # Embeddings
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, is_causal=True)

        # Final norm and LM head
        x = self.norm(x)
        logits = self.lm_head(x)

        if return_loss and targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            return loss

        return logits

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ) -> torch.Tensor:
        """Generate text autoregressively."""
        self.eval()

        for _ in range(max_new_tokens):
            input_ids_crop = input_ids[:, -self.max_seq_len:]
            logits = self(input_ids_crop, return_loss=False)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


# Model creation functions
def create_oelm_v2_tiny(vocab_size: int = 50257, freeze_qk: bool = True, init_method: str = 'orthogonal', **kwargs) -> HeadWiseOrthogonalELMTransformer:
    """Create tiny v2 model for quick experiments."""
    config = dict(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        freeze_qk=freeze_qk,
        init_method=init_method
    )
    config.update(kwargs)
    return HeadWiseOrthogonalELMTransformer(**config)


def create_oelm_v2_small(vocab_size: int = 50257, freeze_qk: bool = True, init_method: str = 'orthogonal', **kwargs) -> HeadWiseOrthogonalELMTransformer:
    """Create small v2 model."""
    config = dict(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1,
        freeze_qk=freeze_qk,
        init_method=init_method
    )
    config.update(kwargs)
    return HeadWiseOrthogonalELMTransformer(**config)


def create_oelm_v2_medium(vocab_size: int = 50257, freeze_qk: bool = True, init_method: str = 'orthogonal', **kwargs) -> HeadWiseOrthogonalELMTransformer:
    """Create medium v2 model."""
    config = dict(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        freeze_qk=freeze_qk,
        init_method=init_method
    )
    config.update(kwargs)
    return HeadWiseOrthogonalELMTransformer(**config)


# Unit tests
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Head-wise Orthogonal ELM Transformer v2")
    print("=" * 60)

    # Test 1: Head-wise orthogonality
    print("\nTest 1: Head-wise Orthogonal Linear")
    head_wise_linear = HeadWiseOrthogonalLinear(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = head_wise_linear(x)
    assert out.shape == (2, 10, 512)

    # Verify head-wise orthogonality
    W = head_wise_linear.weight
    W_heads = W.view(8, 64, 512)
    for h in range(8):
        head = W_heads[h]
        I = head @ head.T
        error = torch.norm(I - torch.eye(64), p='fro')
        assert error < 1e-3, f"Head {h} orthogonality error: {error}"
    print("  ✓ Head-wise orthogonality verified for all heads")

    # Test 2: Multi-head attention
    print("\nTest 2: Head-wise Orthogonal Multi-Head Attention")
    attn = HeadWiseOrthogonalMultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = attn(x)
    assert out.shape == (2, 10, 512)
    print("  ✓ Forward pass successful")

    # Test 3: Full model (OELM-Freeze)
    print("\nTest 3: Full Head-wise Orthogonal ELM Transformer (OELM-Freeze)")
    model = create_oelm_v2_tiny(vocab_size=1000, freeze_qk=True, init_method='orthogonal')
    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))
    loss = model(input_ids, targets)
    assert loss.ndim == 0
    print(f"  ✓ Loss computed: {loss.item():.4f}")

    # Test 4: Full model (OELM-Random for ablation)
    print("\nTest 4: Full Head-wise Orthogonal ELM Transformer (OELM-Random)")
    model_random = create_oelm_v2_tiny(vocab_size=1000, freeze_qk=True, init_method='normal')
    loss_random = model_random(input_ids, targets)
    assert loss_random.ndim == 0
    print(f"  ✓ Loss computed: {loss_random.item():.4f}")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
