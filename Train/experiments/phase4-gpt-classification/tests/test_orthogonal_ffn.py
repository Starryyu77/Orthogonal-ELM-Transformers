"""
正交FFN模块单元测试

测试内容:
1. FrozenOrthogonalLinear的正交性质
2. FrozenOrthogonalLinear的冻结状态
3. OrthogonalFFN的前向传播
4. 参数统计
"""

import sys
import torch
import pytest
from pathlib import Path

# 添加models目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from orthogonal_ffn import FrozenOrthogonalLinear, OrthogonalFFN


class TestFrozenOrthogonalLinear:
    """测试FrozenOrthogonalLinear类。"""

    def test_output_shape(self):
        """测试输出shape正确。"""
        layer = FrozenOrthogonalLinear(512, 2048)
        x = torch.randn(4, 32, 512)
        out = layer(x)
        assert out.shape == (4, 32, 2048)

    def test_frozen_weights(self):
        """测试权重被冻结。"""
        layer = FrozenOrthogonalLinear(512, 2048)
        # 检查requires_grad
        assert not layer.weight.requires_grad
        # 检查是buffer不是parameter
        assert 'weight' in dict(layer.named_buffers()).keys()
        assert 'weight' not in dict(layer.named_parameters()).keys()

    def test_trainable_bias(self):
        """测试偏置是可训练的。"""
        layer = FrozenOrthogonalLinear(512, 2048, bias=True)
        # 偏置应该是parameter
        assert layer.bias is not None
        assert layer.bias.requires_grad

    def test_no_bias(self):
        """测试可以创建无偏置版本。"""
        layer = FrozenOrthogonalLinear(512, 2048, bias=False)
        assert layer.bias is None

    def test_orthogonality_square(self):
        """测试方阵的正交性: W @ W.T = I。"""
        layer = FrozenOrthogonalLinear(512, 512)
        W = layer.weight

        # 检查 W @ W.T ≈ I
        product = W @ W.T
        identity = torch.eye(512)

        max_error = torch.max(torch.abs(product - identity)).item()
        print(f"\n方阵正交性检查: max error = {max_error:.2e}")

        assert max_error < 1e-5, f"Orthogonality check failed: {max_error}"

    def test_orthogonality_tall(self):
        """测试tall矩阵的正交性: W.T @ W = I。"""
        layer = FrozenOrthogonalLinear(512, 2048)  # in < out
        W = layer.weight  # (2048, 512)

        # 对于tall矩阵，检查 W.T @ W ≈ I
        product = W.T @ W  # (512, 512)
        identity = torch.eye(512)

        max_error = torch.max(torch.abs(product - identity)).item()
        print(f"\nTall矩阵正交性检查: max error = {max_error:.2e}")

        assert max_error < 1e-5, f"Orthogonality check failed: {max_error}"

    def test_orthogonality_wide(self):
        """测试wide矩阵的正交性: W @ W.T = I。"""
        layer = FrozenOrthogonalLinear(2048, 512)  # in > out
        W = layer.weight  # (512, 2048)

        # 对于wide矩阵，检查 W @ W.T ≈ I
        product = W @ W.T  # (512, 512)
        identity = torch.eye(512)

        max_error = torch.max(torch.abs(product - identity)).item()
        print(f"\nWide矩阵正交性检查: max error = {max_error:.2e}")

        assert max_error < 1e-5, f"Orthogonality check failed: {max_error}"

    def test_check_orthogonality_method(self):
        """测试check_orthogonality方法。"""
        layer = FrozenOrthogonalLinear(512, 2048)
        # 应该通过检查
        assert layer.check_orthogonality(tolerance=1e-5)

    def test_normal_init(self):
        """测试随机初始化选项。"""
        layer = FrozenOrthogonalLinear(512, 2048, init_method='normal')
        # 随机初始化不应该通过正交检查
        assert not layer.check_orthogonality(tolerance=1e-5)


class TestOrthogonalFFN:
    """测试OrthogonalFFN类。"""

    def test_forward_shape(self):
        """测试前向传播shape正确。"""
        ffn = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=True)
        x = torch.randn(4, 32, 512)
        out = ffn(x)
        assert out.shape == (4, 32, 512)

    def test_frozen_ffn(self):
        """测试冻结FFN。"""
        ffn = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=True)

        # 权重应该是冻结的
        assert not ffn.fc_up.weight.requires_grad
        assert not ffn.fc_down.weight.requires_grad

        # 但偏置应该是可训练的
        assert ffn.fc_up.bias.requires_grad
        assert ffn.fc_down.bias.requires_grad

    def test_trainable_ffn(self):
        """测试非冻结FFN（baseline模式）。"""
        ffn = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=False)

        # 所有参数都应该可训练
        assert ffn.fc_up.weight.requires_grad
        assert ffn.fc_down.weight.requires_grad

    def test_param_stats(self):
        """测试参数统计。"""
        ffn = OrthogonalFFN(d_model=512, d_ff=2048, freeze_ffn=True)
        stats = ffn.get_param_stats()

        # 检查统计结构
        assert 'frozen' in stats
        assert 'trainable' in stats
        assert 'total' in stats
        assert 'frozen_pct' in stats

        # 应该有冻结参数
        assert stats['frozen'] > 0
        # 应该有可训练参数（偏置）
        assert stats['trainable'] > 0
        # 总数正确
        assert stats['total'] == stats['frozen'] + stats['trainable']

    def test_different_dims(self):
        """测试不同维度组合。"""
        configs = [
            (256, 1024),
            (512, 2048),
            (768, 3072),
            (1024, 4096),
        ]

        for d_model, d_ff in configs:
            ffn = OrthogonalFFN(d_model=d_model, d_ff=d_ff)
            x = torch.randn(2, 16, d_model)
            out = ffn(x)
            assert out.shape == (2, 16, d_model)


class TestOrthogonalProperties:
    """测试正交性质的数值稳定性。"""

    def test_multiple_instances(self):
        """测试多个实例产生不同的正交矩阵。"""
        layers = [FrozenOrthogonalLinear(512, 512) for _ in range(5)]

        # 检查正交性
        for i, layer in enumerate(layers):
            W = layer.weight
            product = W @ W.T
            max_error = torch.max(torch.abs(product - torch.eye(512))).item()
            assert max_error < 1e-5, f"Layer {i} failed orthogonality"

        # 检查不同的实例有不同的权重
        weights = [layer.weight for layer in layers]
        for i in range(len(weights)):
            for j in range(i + 1, len(weights)):
                diff = torch.max(torch.abs(weights[i] - weights[j])).item()
                assert diff > 1e-3, "Different instances should have different weights"


class TestGPU:
    """GPU相关测试（仅在CUDA可用时运行）。"""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_forward_on_gpu(self):
        """测试GPU前向传播。"""
        ffn = OrthogonalFFN(d_model=512, d_ff=2048).cuda()
        x = torch.randn(4, 32, 512).cuda()
        out = ffn(x)
        assert out.shape == (4, 32, 512)
        assert out.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_orthogonality_on_gpu(self):
        """测试GPU上的正交性。"""
        layer = FrozenOrthogonalLinear(512, 2048)
        # 移动到GPU
        layer = layer.cuda()

        # 检查正交性（应该在GPU上计算）
        W = layer.weight
        product = W.T @ W
        identity = torch.eye(512, device='cuda')

        max_error = torch.max(torch.abs(product - identity)).item()
        assert max_error < 1e-5


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])
