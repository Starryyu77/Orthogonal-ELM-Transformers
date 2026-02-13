#!/usr/bin/env python3
"""
测试 freeze_qk 参数的不同配置
"""
import sys
sys.path.insert(0, '/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/学术研究/Orthogonal ELM Transformers/Train')

import torch
from models.modeling_oelm import OrthogonalELMTransformer


def test_freeze_config(freeze_qk, name):
    """测试特定freeze配置"""
    print(f"\n{'='*60}")
    print(f"测试: {name} (freeze_qk={freeze_qk})")
    print('='*60)

    torch.manual_seed(42)

    model = OrthogonalELMTransformer(
        vocab_size=1000,
        d_model=256,
        num_layers=2,
        num_heads=4,
        d_ff=1024,
        max_seq_len=512,
        freeze_qk=freeze_qk
    )

    # 统计参数
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())
    total = trainable + buffers

    print(f"\n参数统计:")
    print(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
    print(f"  Frozen (buffers): {buffers:,} ({100*buffers/total:.1f}%)")
    print(f"  Total: {total:,}")

    # 测试训练步骤
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_wq = model.layers[0].self_attn.W_q.weight.clone().detach()

    x = torch.randint(0, 1000, (2, 32))
    y = torch.randint(0, 1000, (2, 32))

    loss = model(x, y)
    loss.backward()
    optimizer.step()

    wq_changed = not torch.allclose(initial_wq, model.layers[0].self_attn.W_q.weight, atol=1e-7)

    print(f"\n训练测试:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  W_q 是否变化: {'是 (可训练)' if wq_changed else '否 (已冻结)'}")

    # 验证一致性
    if freeze_qk and wq_changed:
        print(f"  ⚠ 警告: freeze_qk=True 但 W_q 变化了!")
        return False
    elif not freeze_qk and not wq_changed:
        print(f"  ⚠ 警告: freeze_qk=False 但 W_q 未变化!")
        return False
    else:
        print(f"  ✓ 行为符合预期")
        return True


def main():
    print("="*60)
    print("freeze_qk 参数功能测试")
    print("="*60)

    results = []

    # 测试 freeze_qk=True
    results.append(test_freeze_config(True, "OELM-Freeze"))

    # 测试 freeze_qk=False
    results.append(test_freeze_config(False, "OELM-NoFreeze"))

    print(f"\n{'='*60}")
    print("测试总结")
    print('='*60)

    if all(results):
        print("✓ 所有测试通过!")
        print("freeze_qk 参数工作正常")
        return 0
    else:
        print("✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit(main())
