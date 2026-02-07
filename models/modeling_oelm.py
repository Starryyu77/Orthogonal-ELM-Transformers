"""
Orthogonal ELM Transformer Implementation

This module implements the Orthogonal ELM Attention mechanism where:
- Query (Q) and Key (K) projection matrices are initialized as orthogonal random matrices and frozen
- Value (V) projection and Feed-Forward Networks are trainable

Based on the theoretical foundation that orthogonal projections preserve distances (Isometry property)
and provide stable gradient flow during training.

References:
- Wang et al. "Orthogonal Convolutional Neural Networks" CVPR 2020
- Bansal et al. "Can We Gain More from Orthogonality Regularizations" NeurIPS 2018
- Huang et al. "Extreme Learning Machine" IEEE 2006
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class OrthogonalLinear(nn.Module):
    """
    Linear layer with orthogonal initialization and optional freezing.

    This layer initializes its weight matrix as an orthogonal matrix using QR decomposition
    and can optionally freeze it. The bias remains trainable if specified.

    The orthogonal property ensures: W^T @ W = I (for tall matrices) or W @ W^T = I (for wide matrices)
    This provides the Isometry property: ||Wx|| = ||x||

    Args:
        in_features: Size of input features
        out_features: Size of output features
        bias: Whether to include a bias term (default: True)
        ortho_method: Method for orthogonal initialization ('qr', 'householder', 'cayley')
        freeze: Whether to freeze the weight matrix (default: True)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        ortho_method: str = 'qr',
        freeze: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ortho_method = ortho_method
        self.freeze = freeze

        # Initialize weight as orthogonal matrix
        weight = self._init_orthogonal(in_features, out_features, method=ortho_method)

        # Register as buffer (frozen) or parameter (trainable) based on freeze flag
        if freeze:
            self.register_buffer('weight', weight)
        else:
            self.weight = nn.Parameter(weight.clone())

        # Bias remains trainable
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def _init_orthogonal(
        self, 
        m: int, 
        n: int, 
        method: str = 'qr'
    ) -> torch.Tensor:
        """
        Initialize an orthogonal matrix of shape (m, n) where m >= n.
        
        Args:
            m: Number of rows
            n: Number of columns
            method: Orthogonalization method ('qr', 'householder', 'svd')
            
        Returns:
            Orthogonal matrix W of shape (m, n) satisfying W^T @ W = I_n
        """
        assert m >= n, f"For orthogonal initialization, in_features ({m}) must be >= out_features ({n})"
        
        if method == 'qr':
            # QR decomposition method (recommended for efficiency)
            # Generate random matrix and extract Q from QR decomposition
            A = torch.randn(m, n)
            Q, R = torch.linalg.qr(A, mode='reduced')
            # Adjust signs to ensure deterministic behavior
            signs = torch.sign(torch.diag(R))
            Q = Q * signs.unsqueeze(0)
            return Q
            
        elif method == 'svd':
            # SVD method (more accurate but slower)
            A = torch.randn(m, n)
            U, S, Vh = torch.linalg.svd(A, full_matrices=False)
            # Return the left singular vectors (U)
            return U[:, :n]
            
        elif method == 'householder':
            # Householder reflection method
            # Generate using random reflections
            v = torch.randn(m, n)
            v = F.normalize(v, dim=0)
            Q = torch.eye(m, n)
            for i in range(n):
                Q = Q - 2 * torch.outer(v[:, i], v[:, i]) @ Q
            return Q
            
        else:
            raise ValueError(f"Unknown orthogonal initialization method: {method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the frozen orthogonal transformation.
        
        Args:
            x: Input tensor of shape (..., in_features)
            
        Returns:
            Output tensor of shape (..., out_features)
        """
        return F.linear(x, self.weight, self.bias)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, ' \
               f'bias={self.bias is not None}, ortho_method={self.ortho_method}'


class OrthogonalMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with orthogonal random Q/K projections.

    This module implements the core innovation of Orthogonal ELM Attention:
    - W_q and W_k are initialized as orthogonal random matrices and can be frozen
    - W_v remains trainable

    The attention is computed as: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    where Q = X @ W_q (orthogonal), K = X @ W_k (orthogonal), V = X @ W_v (trainable)

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        ortho_method: Method for orthogonal initialization
        freeze_qk: Whether to freeze Q/K projection matrices (default: True)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ortho_method: str = 'qr',
        freeze_qk: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.freeze_qk = freeze_qk

        # Q and K projections: orthogonal random initialization, optionally frozen
        self.W_q = OrthogonalLinear(d_model, d_model, bias=False, ortho_method=ortho_method, freeze=freeze_qk)
        self.W_k = OrthogonalLinear(d_model, d_model, bias=False, ortho_method=ortho_method, freeze=freeze_qk)
        
        # V projection: normal initialization, trainable
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection: trainable
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
        
        # Verify orthogonal property
        self._verify_orthogonality()
    
    def _verify_orthogonality(self):
        """Verify that Q and K projection matrices are orthogonal."""
        with torch.no_grad():
            W_q = self.W_q.weight  # (d_model, d_model)
            W_k = self.W_k.weight  # (d_model, d_model)
            
            # Check Q orthogonality: W_q^T @ W_q should be close to identity
            I_q = W_q.T @ W_q
            ortho_error_q = torch.norm(I_q - torch.eye(self.d_model, device=W_q.device), p='fro')
            
            # Check K orthogonality
            I_k = W_k.T @ W_k
            ortho_error_k = torch.norm(I_k - torch.eye(self.d_model, device=W_k.device), p='fro')
            
            # These should be very small (numerical precision level)
            if ortho_error_q > 1e-4 or ortho_error_k > 1e-4:
                print(f"Warning: Orthogonality error Q={ortho_error_q:.6f}, K={ortho_error_k:.6f}")
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of orthogonal multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking for autoregressive models
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V projections
        # Q and K use frozen orthogonal weights
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
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, seq_len, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, V)  # (batch, num_heads, seq_len, d_k)
        
        # Concatenate heads: (batch, seq_len, d_model)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Final linear projection
        out = self.W_o(out)
        
        return out


class OrthogonalTransformerLayer(nn.Module):
    """
    Single Transformer layer with Orthogonal ELM Attention.

    Architecture:
    1. Orthogonal Multi-Head Attention (with pre-norm)
    2. Feed-Forward Network (with pre-norm)
    3. Residual connections around both sublayers

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
        ortho_method: Method for orthogonal initialization
        freeze_qk: Whether to freeze Q/K projection matrices (default: True)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        ortho_method: str = 'qr',
        freeze_qk: bool = True
    ):
        super().__init__()

        # Orthogonal multi-head attention
        self.self_attn = OrthogonalMultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            ortho_method=ortho_method,
            freeze_qk=freeze_qk
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of transformer layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Self-attention sublayer with residual (pre-norm)
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask, is_causal=is_causal)
        x = x + self.dropout(attn_out)
        
        # Feed-forward sublayer with residual (pre-norm)
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class OrthogonalELMTransformer(nn.Module):
    """
    Complete Transformer model with Orthogonal ELM Attention.

    This model stacks multiple OrthogonalTransformerLayer layers and adds:
    - Token embeddings
    - Positional embeddings
    - Output language modeling head

    The key innovation is that all Q/K projection matrices across all layers
    are initialized as orthogonal random matrices and can be frozen,
    significantly reducing trainable parameters while maintaining expressiveness.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        ortho_method: Method for orthogonal initialization
        freeze_qk: Whether to freeze Q/K projection matrices (default: True)
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
        ortho_method: str = 'qr',
        freeze_qk: bool = True
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.freeze_qk = freeze_qk

        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings (learnable)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer layers with orthogonal attention
        self.layers = nn.ModuleList([
            OrthogonalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                dropout=dropout,
                ortho_method=ortho_method,
                freeze_qk=freeze_qk
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights between token embedding and LM head
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        
        # Initialize parameters
        self.apply(self._init_weights)
        
        # Print model statistics
        self._print_model_info()
    
    def _init_weights(self, module):
        """Initialize weights (except orthogonal layers which are already initialized)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _print_model_info(self):
        """Print model parameter statistics."""
        # Count trainable parameters (nn.Parameter with requires_grad=True)
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count frozen parameters from buffers (frozen Q/K weights)
        buffer_params = sum(b.numel() for b in self.buffers())

        # Total parameters including buffers
        all_params = trainable_params + buffer_params

        # Calculate frozen percentage
        frozen_pct = 100 * buffer_params / all_params if all_params > 0 else 0
        trainable_pct = 100 * trainable_params / all_params if all_params > 0 else 0

        print(f"OrthogonalELMTransformer:")
        print(f"  Total parameters: {all_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_pct:.1f}%)")
        print(f"  Frozen parameters: {buffer_params:,} ({frozen_pct:.1f}%)")
        print(f"  Q/K frozen: {self.freeze_qk}")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            targets: Target token IDs for loss computation (optional)
            return_loss: Whether to compute and return loss
            
        Returns:
            If return_loss=True: loss scalar
            If return_loss=False: logits of shape (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create position indices
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token + positional embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, is_causal=True)
        
        # Final layer norm
        x = self.norm(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        if return_loss and targets is not None:
            # Compute cross-entropy loss
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
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Initial token IDs of shape (batch_size, seq_len)
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling (optional)
            
        Returns:
            Generated token IDs of shape (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_crop = input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self(input_ids_crop, return_loss=False)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids


# Model configuration functions
def create_oelm_tiny(vocab_size: int = 50257, freeze_qk: bool = True, **kwargs) -> OrthogonalELMTransformer:
    """Create tiny OELM model for quick experiments."""
    config = dict(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1,
        freeze_qk=freeze_qk
    )
    config.update(kwargs)
    return OrthogonalELMTransformer(**config)


def create_oelm_small(vocab_size: int = 50257, freeze_qk: bool = True, **kwargs) -> OrthogonalELMTransformer:
    """Create small OELM model (~30M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_len=1024,
        dropout=0.1,
        freeze_qk=freeze_qk
    )
    config.update(kwargs)
    return OrthogonalELMTransformer(**config)


def create_oelm_medium(vocab_size: int = 50257, freeze_qk: bool = True, **kwargs) -> OrthogonalELMTransformer:
    """Create medium OELM model (~90M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1,
        freeze_qk=freeze_qk
    )
    config.update(kwargs)
    return OrthogonalELMTransformer(**config)


def create_oelm_large(vocab_size: int = 50257, freeze_qk: bool = True, **kwargs) -> OrthogonalELMTransformer:
    """Create large OELM model (~260M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        max_seq_len=1024,
        dropout=0.1,
        freeze_qk=freeze_qk
    )
    config.update(kwargs)
    return OrthogonalELMTransformer(**config)


# Unit tests
if __name__ == "__main__":
    print("Running Orthogonal ELM Transformer tests...\n")
    
    # Test 1: OrthogonalLinear
    print("Test 1: OrthogonalLinear")
    ortho_linear = OrthogonalLinear(512, 512)
    x = torch.randn(2, 10, 512)
    out = ortho_linear(x)
    assert out.shape == (2, 10, 512), f"Expected (2, 10, 512), got {out.shape}"
    
    # Verify orthogonality
    W = ortho_linear.weight
    I = W.T @ W
    ortho_error = torch.norm(I - torch.eye(512), p='fro')
    print(f"  Orthogonality error: {ortho_error:.6f} (should be < 1e-4)")
    assert ortho_error < 1e-4, f"Orthogonality error too large: {ortho_error}"
    print("  ✓ Passed\n")
    
    # Test 2: OrthogonalMultiHeadAttention
    print("Test 2: OrthogonalMultiHeadAttention")
    attn = OrthogonalMultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = attn(x)
    assert out.shape == (2, 10, 512), f"Expected (2, 10, 512), got {out.shape}"
    print("  ✓ Passed\n")
    
    # Test 3: OrthogonalTransformerLayer
    print("Test 3: OrthogonalTransformerLayer")
    layer = OrthogonalTransformerLayer(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = layer(x)
    assert out.shape == (2, 10, 512), f"Expected (2, 10, 512), got {out.shape}"
    print("  ✓ Passed\n")
    
    # Test 4: OrthogonalELMTransformer
    print("Test 4: OrthogonalELMTransformer")
    model = create_oelm_tiny(vocab_size=1000)
    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))
    loss = model(input_ids, targets)
    assert loss.ndim == 0, "Loss should be a scalar"
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ Passed\n")
    
    # Test 5: Generation
    print("Test 5: Generation")
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=10)
    assert generated.shape == (1, 15), f"Expected (1, 15), got {generated.shape}"
    print(f"  Generated shape: {generated.shape}")
    print("  ✓ Passed\n")
    
    print("All tests passed! ✓")
