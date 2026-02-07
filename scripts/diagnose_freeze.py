#!/usr/bin/env python3
"""
诊断脚本：验证OELM冻结机制状态
运行此脚本检查Q/K矩阵是否真正被冻结
"""
import sys
import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.insert(0, '/Users/starryyu/Library/Mobile Documents/com~apple~CloudDocs/学术研究/Orthogonal ELM Transformers/Train')

from models.modeling_oelm import OrthogonalELMTransformer, OrthogonalLinear


def diagnose_model():
    print("=" * 70)
    print("OELM 冻结机制诊断报告")
    print("=" * 70)

    # 创建模型
    print("\n[1] 创建OELM模型...")
    model = OrthogonalELMTransformer(
        vocab_size=50257,
        d_model=512,
        num_layers=6,
        num_heads=8,
        d_ff=2048,
        max_seq_len=512
    )
    print("   ✓ 模型创建成功")

    # 1. 检查所有参数
    print("\n[2] 参数类型检查 (所有nn.Parameter):")
    print("-" * 70)

    qk_params_found = []
    for name, param in model.named_parameters():
        if 'W_q' in name or 'W_k' in name:
            qk_params_found.append((name, param))
            print(f"   Parameter: {name}")
            print(f"   Shape: {param.shape}")
            print(f"   requires_grad: {param.requires_grad}")
            print()

    if not qk_params_found:
        print("   ✓ 未发现W_q/W_k的nn.Parameter (说明可能是buffer)")
    else:
        print(f"   ⚠ 发现{len(qk_params_found)}个W_q/W_k的nn.Parameter")

    # 2. 检查所有buffers
    print("\n[3] Buffer检查 (冻结的参数):")
    print("-" * 70)

    buffers_found = []
    for name, buffer in model.named_buffers():
        buffers_found.append((name, buffer))
        if 'W_q' in name or 'W_k' in name:
            print(f"   Buffer: {name}")
            print(f"   Shape: {buffer.shape}")
            print(f"   dtype: {buffer.dtype}")
            print()

    if not buffers_found:
        print("   ⚠ 未发现任何buffers")
    else:
        print(f"   ✓ 共发现{len(buffers_found)}个buffers")

    # 3. 统计信息
    print("\n[4] 参数统计:")
    print("-" * 70)

    # 传统参数统计
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_from_params = total_params - trainable_params

    # Buffer统计
    buffer_params = sum(b.numel() for b in model.buffers())

    # 总参数量
    all_params = total_params + buffer_params

    print(f"   nn.Parameter总参数量: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)")
    print(f"   冻结参数(来自nn.Parameter): {frozen_from_params:,}")
    print(f"   Buffer参数量(冻结): {buffer_params:,}")
    print(f"   实际总参数量: {all_params:,}")

    # 4. 检查第一层Attention的详细信息
    print("\n[5] OrthogonalLinear详细检查 (第一层Attention):")
    print("-" * 70)

    first_layer = model.layers[0]
    w_q = first_layer.self_attn.W_q
    w_k = first_layer.self_attn.W_k
    w_v = first_layer.self_attn.W_v

    print(f"\n   W_q (Query投影):")
    print(f"      类型: {type(w_q.weight)}")
    print(f"      是否为nn.Parameter: {isinstance(w_q.weight, nn.Parameter)}")
    print(f"      是否为buffer: {name in dict(model.named_buffers())}")
    print(f"      requires_grad: {w_q.weight.requires_grad}")

    # 验证正交性
    W = w_q.weight
    I = W.T @ W
    ortho_error = torch.norm(I - torch.eye(W.shape[1]), p='fro')
    print(f"      正交性误差: {ortho_error:.6f}")

    print(f"\n   W_k (Key投影):")
    print(f"      类型: {type(w_k.weight)}")
    print(f"      是否为nn.Parameter: {isinstance(w_k.weight, nn.Parameter)}")
    print(f"      requires_grad: {w_k.weight.requires_grad}")

    print(f"\n   W_v (Value投影 - 对比):")
    print(f"      类型: {type(w_v.weight)}")
    print(f"      是否为nn.Parameter: {isinstance(w_v.weight, nn.Parameter)}")
    print(f"      requires_grad: {w_v.weight.requires_grad}")

    # 5. 模拟训练步骤验证权重是否变化
    print("\n[6] 训练步骤验证 (检查权重是否真正冻结):")
    print("-" * 70)

    # 保存初始权重
    initial_w_q = w_q.weight.clone().detach()
    initial_w_k = w_k.weight.clone().detach()
    initial_w_v = w_v.weight.clone().detach()

    # 创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 模拟前向传播
    batch_size = 2
    seq_len = 64
    input_ids = torch.randint(0, 50257, (batch_size, seq_len))
    targets = torch.randint(0, 50257, (batch_size, seq_len))

    print("   执行前向传播...")
    try:
        loss = model(input_ids, targets)
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   前向传播出错: {e}")
        return

    print("   执行反向传播...")
    loss.backward()

    # 检查梯度
    print("\n   梯度检查:")
    if w_q.weight.grad is not None:
        grad_norm = w_q.weight.grad.norm().item()
        print(f"      W_q梯度范数: {grad_norm:.6f}")
    else:
        print(f"      W_q梯度: None (已冻结)")

    if w_k.weight.grad is not None:
        grad_norm = w_k.weight.grad.norm().item()
        print(f"      W_k梯度范数: {grad_norm:.6f}")
    else:
        print(f"      W_k梯度: None (已冻结)")

    if w_v.weight.grad is not None:
        grad_norm = w_v.weight.grad.norm().item()
        print(f"      W_v梯度范数: {grad_norm:.6f}")
    else:
        print(f"      W_v梯度: None")

    print("   执行优化器步骤...")
    optimizer.step()

    # 检查权重变化
    print("\n   权重变化检查:")
    w_q_changed = not torch.allclose(initial_w_q, w_q.weight, atol=1e-7)
    w_k_changed = not torch.allclose(initial_w_k, w_k.weight, atol=1e-7)
    w_v_changed = not torch.allclose(initial_w_v, w_v.weight, atol=1e-7)

    print(f"      W_q (Query): {'⚠ 已变化 (未冻结!)' if w_q_changed else '✓ 未变化 (已冻结)'}")
    print(f"      W_k (Key): {'⚠ 已变化 (未冻结!)' if w_k_changed else '✓ 未变化 (已冻结)'}")
    print(f"      W_v (Value): {'⚠ 已变化' if w_v_changed else '✓ 未变化'}")

    # 6. 最终结论
    print("\n" + "=" * 70)
    print("诊断结论")
    print("=" * 70)

    if not w_q_changed and not w_k_changed:
        print("✓ Q/K矩阵冻结机制工作正常")
        print(f"  - 冻结参数量: {buffer_params:,} ({100*buffer_params/all_params:.1f}%)")
        print(f"  - 节省的可训练参数: ~{buffer_params:,}")
        return True
    else:
        print("⚠ Q/K矩阵冻结机制存在问题!")
        print(f"  - W_q状态: {'已冻结' if not w_q_changed else '未冻结'}")
        print(f"  - W_k状态: {'已冻结' if not w_k_changed else '未冻结'}")
        print("\n可能原因:")
        print("  1. OrthogonalLinear未正确使用register_buffer")
        print("  2. 检查点加载时覆盖了冻结参数")
        print("  3. 模型初始化后手动修改了requires_grad")
        return False


if __name__ == "__main__":
    # 设置随机种子确保可重复
    torch.manual_seed(42)

    success = diagnose_model()

    print("\n" + "=" * 70)
    if success:
        print("诊断完成: 冻结机制正常工作 ✓")
        sys.exit(0)
    else:
        print("诊断完成: 发现冻结机制问题 ⚠")
        sys.exit(1)
