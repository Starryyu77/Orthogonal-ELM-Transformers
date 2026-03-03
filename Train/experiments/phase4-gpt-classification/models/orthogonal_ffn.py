"""
正交FFN模块 (Orthogonal FFN)

提供冻结的正交线性变换，用于FFN的升维和降维矩阵。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FrozenOrthogonalLinear(nn.Module):
    """
    冻结的正交线性层。

    使用torch.nn.init.orthogonal_进行初始化，然后冻结权重。
    支持任意形状的矩阵（不一定是方阵）。

    Args:
        in_features: 输入特征维度
        out_features: 输出特征维度
        bias: 是否使用偏置（默认True）
        init_method: 初始化方法，'orthogonal' 或 'normal'
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: str = 'orthogonal'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.init_method = init_method

        # 初始化权重
        weight = torch.empty(out_features, in_features)
        if init_method == 'orthogonal':
            # 使用PyTorch内置的正交初始化
            nn.init.orthogonal_(weight)
        elif init_method == 'normal':
            # 标准正态初始化（用于消融实验）
            nn.init.normal_(weight, mean=0.0, std=0.02)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")

        # 注册为buffer（冻结，不计算梯度）
        self.register_buffer('weight', weight)

        # 偏置是可训练的
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return F.linear(x, self.weight, self.bias)

    def check_orthogonality(self, tolerance: float = 1e-5) -> bool:
        """
        验证正交性质。

        对于 tall 矩阵 (out > in): W^T @ W ≈ I_in
        对于 wide 矩阵 (out < in): W @ W^T ≈ I_out
        对于方阵 (out == in): W @ W^T ≈ I

        Args:
            tolerance: 数值容差

        Returns:
            是否通过正交性检查
        """
        with torch.no_grad():
            W = self.weight  # (out_features, in_features)

            if self.out_features >= self.in_features:
                # Tall or square: check W^T @ W ≈ I
                product = W.T @ W  # (in_features, in_features)
                identity = torch.eye(self.in_features, device=W.device, dtype=W.dtype)
                max_error = torch.max(torch.abs(product - identity)).item()

                if max_error > tolerance:
                    print(f"⚠️ Orthogonality check: W^T @ W max error = {max_error:.2e}")
                    return False
            else:
                # Wide: check W @ W^T ≈ I
                product = W @ W.T  # (out_features, out_features)
                identity = torch.eye(self.out_features, device=W.device, dtype=W.dtype)
                max_error = torch.max(torch.abs(product - identity)).item()

                if max_error > tolerance:
                    print(f"⚠️ Orthogonality check: W @ W^T max error = {max_error:.2e}")
                    return False

            return True

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, " \
               f"bias={self.bias is not None}, init_method={self.init_method}, frozen=True"


class OrthogonalFFN(nn.Module):
    """
    使用正交矩阵的Feed-Forward Network。

    升维（d_model -> d_ff）和降维（d_ff -> d_model）矩阵
    都使用正交初始化并冻结。

    Args:
        d_model: 模型维度
        d_ff: FFN中间维度
        dropout: Dropout概率
        freeze_ffn: 是否冻结FFN权重（False时使用标准可训练层）
        init_method: 初始化方法
        activation: 激活函数类型，默认GELU
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        freeze_ffn: bool = True,
        init_method: str = 'orthogonal',
        activation: str = 'gelu'
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.freeze_ffn = freeze_ffn

        if freeze_ffn:
            # 使用冻结的正交矩阵
            self.fc_up = FrozenOrthogonalLinear(
                d_model, d_ff, bias=True, init_method=init_method
            )
            self.fc_down = FrozenOrthogonalLinear(
                d_ff, d_model, bias=True, init_method=init_method
            )
        else:
            # 标准可训练层（用于baseline对比）
            self.fc_up = nn.Linear(d_model, d_ff)
            self.fc_down = nn.Linear(d_ff, d_model)

        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        self.dropout = nn.Dropout(dropout)

        # 初始化后检查正交性
        if freeze_ffn and init_method == 'orthogonal':
            self._verify_init()

    def _verify_init(self):
        """验证正交初始化正确。"""
        up_ok = self.fc_up.check_orthogonality()
        down_ok = self.fc_down.check_orthogonality()

        if up_ok and down_ok:
            print(f"✓ FFN orthogonal initialization verified")
        else:
            print(f"⚠️ FFN orthogonal initialization check failed")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        Args:
            x: 输入张量 (batch_size, seq_len, d_model)

        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        # 升维
        x = self.fc_up(x)
        x = self.activation(x)
        x = self.dropout(x)

        # 降维
        x = self.fc_down(x)
        x = self.dropout(x)

        return x

    def get_param_stats(self) -> dict:
        """
        获取参数统计信息。

        Returns:
            包含冻结和可训练参数数量的字典
        """
        # 可训练参数（包括偏置和dropout）
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # 冻结参数（buffer中的权重）
        frozen = sum(b.numel() for b in self.buffers())

        # 如果FFN没有冻结，所有参数都是可训练的
        if not self.freeze_ffn:
            frozen = 0
            trainable = sum(p.numel() for p in self.parameters())

        return {
            'frozen': frozen,
            'trainable': trainable,
            'total': frozen + trainable,
            'frozen_pct': frozen / (frozen + trainable) * 100 if (frozen + trainable) > 0 else 0
        }


# 用于测试的辅助函数
def test_orthogonal_ffn():
    """简单的自测函数。"""
    print("Testing OrthogonalFFN...")

    # 测试冻结版本
    ffn_frozen = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=True)
    x = torch.randn(2, 16, 512)
    out = ffn_frozen(x)
    assert out.shape == (2, 16, 512), f"Output shape mismatch: {out.shape}"

    stats = ffn_frozen.get_param_stats()
    print(f"  Frozen FFN stats: {stats}")
    assert stats['frozen'] > 0, "Should have frozen parameters"
    assert stats['trainable'] > 0, "Should have trainable parameters (bias)"

    # 测试非冻结版本
    ffn_train = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=False)
    out = ffn_train(x)
    assert out.shape == (2, 16, 512)

    # 测试正交性
    orth_ok = ffn_frozen.fc_up.check_orthogonality()
    print(f"  Orthogonality check: {'PASS' if orth_ok else 'FAIL'}")

    print("All tests passed!")


if __name__ == "__main__":
    test_orthogonal_ffn()
