"""
Standard GPT Model Implementation

This module implements a standard GPT (Generative Pre-trained Transformer) model
for comparison with the Orthogonal ELM Transformer.

All attention parameters are trainable in this implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Attention (all parameters trainable).
    
    This is the standard attention mechanism used in GPT-2 and similar models.
    All projection matrices (W_q, W_k, W_v, W_o) are trainable.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        d_model: int, 
        num_heads: int = 8, 
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Q, K, V projections: all trainable
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection: trainable
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of multi-head attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            is_causal: Whether to apply causal masking
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V projections (all trainable)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if is_causal:
            causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply attention to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.W_o(out)
        
        return out


class GPTTransformerLayer(nn.Module):
    """
    Standard GPT Transformer layer.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        dropout: Dropout probability
    """
    
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
        mask: Optional[torch.Tensor] = None,
        is_causal: bool = True
    ) -> torch.Tensor:
        """Forward pass of transformer layer."""
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out = self.self_attn(normed, mask=mask, is_causal=is_causal)
        x = x + self.dropout(attn_out)
        
        # Feed-forward with residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)
        x = x + ffn_out
        
        return x


class GPT(nn.Module):
    """
    Standard GPT (Generative Pre-trained Transformer) model.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of transformer layers
        num_heads: Number of attention heads per layer
        d_ff: Feed-forward hidden dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional embeddings
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # Transformer layers
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
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.token_embedding.weight
        
        self.dropout = nn.Dropout(dropout)
        self.max_seq_len = max_seq_len
        
        self.apply(self._init_weights)
        self._print_model_info()
    
    def _init_weights(self, module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def _print_model_info(self):
        """Print model parameter statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"GPT Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,} (100.0%)")
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True
    ) -> torch.Tensor:
        """Forward pass of the model."""
        batch_size, seq_len = input_ids.shape
        
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(x, is_causal=True)
        
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


# GPT-2 style model configurations
def create_gpt_tiny(vocab_size: int = 50257, **kwargs) -> GPT:
    """Create tiny GPT model for quick experiments."""
    config = dict(
        vocab_size=vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=512,
        dropout=0.1
    )
    config.update(kwargs)
    return GPT(**config)


def create_gpt2_small(vocab_size: int = 50257, **kwargs) -> GPT:
    """Create GPT-2 Small model (~124M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072,
        max_seq_len=1024,
        dropout=0.1
    )
    config.update(kwargs)
    return GPT(**config)


def create_gpt2_medium(vocab_size: int = 50257, **kwargs) -> GPT:
    """Create GPT-2 Medium model (~350M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4096,
        max_seq_len=1024,
        dropout=0.1
    )
    config.update(kwargs)
    return GPT(**config)


def create_gpt2_large(vocab_size: int = 50257, **kwargs) -> GPT:
    """Create GPT-2 Large model (~774M parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=1280,
        num_layers=36,
        num_heads=20,
        d_ff=5120,
        max_seq_len=1024,
        dropout=0.1
    )
    config.update(kwargs)
    return GPT(**config)


def create_gpt2_xl(vocab_size: int = 50257, **kwargs) -> GPT:
    """Create GPT-2 XL model (~1.5B parameters)."""
    config = dict(
        vocab_size=vocab_size,
        d_model=1600,
        num_layers=48,
        num_heads=25,
        d_ff=6400,
        max_seq_len=1024,
        dropout=0.1
    )
    config.update(kwargs)
    return GPT(**config)


if __name__ == "__main__":
    print("Running GPT model tests...\n")
    
    # Test 1: MultiHeadAttention
    print("Test 1: MultiHeadAttention")
    attn = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = attn(x)
    assert out.shape == (2, 10, 512)
    print("  ✓ Passed\n")
    
    # Test 2: GPTTransformerLayer
    print("Test 2: GPTTransformerLayer")
    layer = GPTTransformerLayer(d_model=512, num_heads=8)
    x = torch.randn(2, 10, 512)
    out = layer(x)
    assert out.shape == (2, 10, 512)
    print("  ✓ Passed\n")
    
    # Test 3: GPT model
    print("Test 3: GPT Model")
    model = create_gpt_tiny(vocab_size=1000)
    input_ids = torch.randint(0, 1000, (2, 10))
    targets = torch.randint(0, 1000, (2, 10))
    loss = model(input_ids, targets)
    assert loss.ndim == 0
    print(f"  Loss: {loss.item():.4f}")
    print("  ✓ Passed\n")
    
    # Test 4: Generation
    print("Test 4: Generation")
    model.eval()
    input_ids = torch.randint(0, 1000, (1, 5))
    generated = model.generate(input_ids, max_new_tokens=10)
    assert generated.shape == (1, 15)
    print(f"  Generated shape: {generated.shape}")
    print("  ✓ Passed\n")
    
    print("All tests passed! ✓")
