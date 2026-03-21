from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OELMConfig:
    vocab_size: int = 50257
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    max_seq_len: int = 1024
    dropout: float = 0.1
    freeze_qk: bool = False
    freeze_ffn: bool = False
    init_method: str = "normal"
    model_size: str = "small"

    @classmethod
    def for_model_size(
        cls,
        model_size: str,
        vocab_size: int,
        max_seq_len: int,
        freeze_qk: bool,
        freeze_ffn: bool,
    ) -> "OELMConfig":
        if model_size == "mini":
            return cls(
                vocab_size=vocab_size,
                d_model=512,
                num_layers=6,
                num_heads=8,
                d_ff=2048,
                max_seq_len=max_seq_len,
                freeze_qk=freeze_qk,
                freeze_ffn=freeze_ffn,
                init_method="orthogonal" if freeze_qk or freeze_ffn else "normal",
                model_size="mini",
            )
        if model_size == "small":
            return cls(
                vocab_size=vocab_size,
                d_model=768,
                num_layers=12,
                num_heads=12,
                d_ff=3072,
                max_seq_len=max_seq_len,
                freeze_qk=freeze_qk,
                freeze_ffn=freeze_ffn,
                init_method="orthogonal" if freeze_qk or freeze_ffn else "normal",
                model_size="small",
            )
        raise ValueError(f"Unsupported model_size: {model_size}")


class FrozenLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        init_method: str = "orthogonal",
    ) -> None:
        super().__init__()
        weight = torch.empty(out_features, in_features)
        if init_method == "orthogonal":
            nn.init.orthogonal_(weight)
        elif init_method == "normal":
            nn.init.normal_(weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unsupported init_method: {init_method}")
        self.register_buffer("weight", weight)
        if bias:
            bias_tensor = torch.zeros(out_features)
            self.register_buffer("bias", bias_tensor)
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class HeadWiseOrthogonalLinear(nn.Module):
    def __init__(self, d_model: int, num_heads: int, *, bias: bool = False) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        head_dim = d_model // num_heads
        head_weights = []
        for _ in range(num_heads):
            matrix = torch.randn(d_model, head_dim)
            q_factor, r_factor = torch.linalg.qr(matrix, mode="reduced")
            weights = q_factor.T
            signs = torch.sign(torch.diag(r_factor))
            weights = weights * signs.unsqueeze(1)
            head_weights.append(weights)
        full_weight = torch.stack(head_weights, dim=0).view(d_model, d_model).contiguous()
        self.register_buffer("weight", full_weight)
        if bias:
            self.register_buffer("bias", torch.zeros(d_model))
        else:
            self.register_buffer("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class OELMMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        dropout: float,
        freeze_qk: bool,
        init_method: str,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        if freeze_qk:
            self.W_q = HeadWiseOrthogonalLinear(d_model, num_heads, bias=False)
            self.W_k = HeadWiseOrthogonalLinear(d_model, num_heads, bias=False)
        else:
            self.W_q = nn.Linear(d_model, d_model, bias=False)
            self.W_k = nn.Linear(d_model, d_model, bias=False)
            if init_method == "normal":
                nn.init.normal_(self.W_q.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.W_k.weight, mean=0.0, std=0.02)

        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        if init_method == "normal":
            nn.init.normal_(self.W_v.weight, mean=0.0, std=0.02)
            nn.init.normal_(self.W_o.weight, mean=0.0, std=0.02)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        query = self.W_q(x)
        key = self.W_k(x)
        value = self.W_v(x)

        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) / self.scale
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            key_padding_mask = attention_mask[:, None, None, :].eq(0)
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_o(out)


class OELMFFN(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        dropout: float,
        freeze_ffn: bool,
        init_method: str,
    ) -> None:
        super().__init__()
        if freeze_ffn:
            self.up_proj = FrozenLinear(d_model, d_ff, bias=True, init_method="orthogonal")
            self.down_proj = FrozenLinear(d_ff, d_model, bias=True, init_method="orthogonal")
        else:
            self.up_proj = nn.Linear(d_model, d_ff)
            self.down_proj = nn.Linear(d_ff, d_model)
            if init_method == "normal":
                nn.init.normal_(self.up_proj.weight, mean=0.0, std=0.02)
                nn.init.normal_(self.down_proj.weight, mean=0.0, std=0.02)
                nn.init.zeros_(self.up_proj.bias)
                nn.init.zeros_(self.down_proj.bias)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.down_proj(x)
        x = self.dropout(x)
        return x


class OELMTransformerBlock(nn.Module):
    def __init__(
        self,
        config: OELMConfig,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = OELMMultiHeadAttention(
            config.d_model,
            config.num_heads,
            dropout=config.dropout,
            freeze_qk=config.freeze_qk,
            init_method=config.init_method,
        )
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = OELMFFN(
            config.d_model,
            config.d_ff,
            dropout=config.dropout,
            freeze_ffn=config.freeze_ffn,
            init_method=config.init_method,
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.ln1(x), attention_mask=attention_mask))
        x = x + self.ffn(self.ln2(x))
        return x


class OELMForLanguageModeling(nn.Module):
    def __init__(self, config: OELMConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.blocks = nn.ModuleList([OELMTransformerBlock(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds model max_seq_len {self.config.max_seq_len}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)
        return self.ln_f(x)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        hidden_states = self.forward_features(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return {"logits": logits, "loss": loss}

    def save_checkpoint(
        self,
        output_dir: str | Path,
        tokenizer: Any,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        resolved = Path(output_dir)
        resolved.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), resolved / "model_state.pt")
        metadata = asdict(self.config)
        if extra_metadata:
            metadata.update(extra_metadata)
        (resolved / "model_config.json").write_text(json.dumps(metadata, indent=2))
        tokenizer.save_pretrained(resolved)


def create_model(
    *,
    model_size: str,
    method: str,
    vocab_size: int,
    max_seq_len: int,
) -> OELMForLanguageModeling:
    if method not in {"baseline", "qk_only", "qk_ffn"}:
        raise ValueError(f"Unsupported method: {method}")
    freeze_qk = method in {"qk_only", "qk_ffn"}
    freeze_ffn = method == "qk_ffn"
    config = OELMConfig.for_model_size(
        model_size=model_size,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        freeze_qk=freeze_qk,
        freeze_ffn=freeze_ffn,
    )
    return OELMForLanguageModeling(config)


def load_checkpoint(
    checkpoint_dir: str | Path,
    *,
    device: torch.device | None = None,
) -> tuple[OELMForLanguageModeling, dict[str, Any]]:
    resolved = Path(checkpoint_dir)
    metadata = json.loads((resolved / "model_config.json").read_text())
    config = OELMConfig(
        vocab_size=metadata["vocab_size"],
        d_model=metadata["d_model"],
        num_layers=metadata["num_layers"],
        num_heads=metadata["num_heads"],
        d_ff=metadata["d_ff"],
        max_seq_len=metadata["max_seq_len"],
        dropout=metadata.get("dropout", 0.1),
        freeze_qk=metadata["freeze_qk"],
        freeze_ffn=metadata["freeze_ffn"],
        init_method=metadata.get("init_method", "normal"),
        model_size=metadata.get("model_size", "small"),
    )
    model = OELMForLanguageModeling(config)
    state = torch.load(resolved / "model_state.pt", map_location=device or "cpu")
    model.load_state_dict(state)
    if device is not None:
        model.to(device)
    return model, metadata
