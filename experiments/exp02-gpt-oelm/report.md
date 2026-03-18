# Phase 2 实验报告: GPT OELM 移植

> 将 BERT 上验证成功的 OELM 方法移植到 GPT 架构

---

## 1. 实验目标

将Phase 1在BERT上验证成功的分头正交初始化方法移植到GPT架构。

### 1.1 背景

**之前GPT+全局正交失败**:
- 方法: 对GPT的Q/K进行全局QR分解
- 结果: PPL 4.14 → 4.94 (+19%)
- 原因: 全局正交破坏了多头结构

**解决方案: 分头正交**:
- 方法: 将Q/K重塑为[num_heads, head_dim, hidden_dim]
- 每个head独立QR分解
- 预期: 保留多头表达能力

---

## 2. 技术实现

### 2.1 核心代码

**文件**: `models/modeling_oelm_v2.py`

```python
class OELMMultiHeadAttention(nn.Module):
    """GPT OELM 注意力层 - 分头正交初始化"""

    def __init__(self, d_model, num_heads):
        # 标准线性层
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 分头正交初始化
        self._init_orthogonal_heads()

        # 冻结 Q/K
        self._freeze_qk()
```

### 2.2 分头正交初始化

```python
def _init_orthogonal_heads(self):
    """分头正交初始化"""
    d_head = self.d_model // self.num_heads
    for h in range(self.num_heads):
        # 每个头单独正交初始化
        A = torch.randn(self.d_model, d_head)
        Q, R = torch.linalg.qr(A, mode='reduced')
        # 使用Q作为初始化权重
```

---

## 3. 模型配置

### 3.1 GPT Medium-512

| 配置 | 值 |
|------|-----|
| d_model | 512 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 2048 |
| seq_len | 512 |
| 总参数量 | ~82M |

---

## 4. 移植验证

### 4.1 功能验证

| 验证项 | 状态 | 说明 |
|--------|------|------|
| 正交初始化正确 | ✅ | 分头QR分解输出正交矩阵 |
| Q/K冻结正确 | ✅ | requires_grad=False |
| V/O可训练 | ✅ | requires_grad=True |
| 前向传播正常 | ✅ | 输出shape正确 |
| 反向传播正常 | ✅ | 梯度正常传播 |

### 4.2 训练验证

| 验证项 | 状态 | 说明 |
|--------|------|------|
| 训练稳定 | ✅ | 无NaN/Inf |
| Loss下降 | ✅ | 正常收敛 |
| PPL合理 | ✅ | 在预期范围内 |
| 计时功能 | ✅ | CUDA同步计时准确 |

---

## 5. 实验结果

### 5.1 TinyStories初步结果

| 指标 | Baseline | OELM-Freeze | 对比 |
|------|----------|-------------|------|
| Final PPL | 4.27 | 4.69 | **+9.8%** ❌ |
| 训练时间 | 4h 31m | 4h 32m | 相同 |
| 每步时间 | 0.184s | 0.184s | 相同 |

### 5.2 结果分析

**技术移植**: ✅ 成功
- 分头正交实现正确
- 训练流程稳定

**性能目标**: ❌ 未达成
- 目标: PPL ≤ 4.48 (Baseline × 1.05)
- 实际: PPL 4.69 (Baseline × 1.098)
- 差距: 超出5%目标

---

## 6. 关键发现

### 6.1 分头正交 vs 全局正交

| 方法 | PPL结果 | 状态 |
|------|---------|------|
| 全局正交 | +19% | ❌ 失败 |
| 分头正交 | +9.8% | ⚠️ 可用但不够理想 |

**结论**: 分头正交显著优于全局正交，但仍未达目标

### 6.2 BERT vs GPT对比

| 架构 | 任务 | OELM效果 | 原因 |
|------|------|----------|------|
| BERT | 分类 | ✅ +1.08% | 注意力模式稳定 |
| GPT | 生成 | ❌ -9.8% | 需要动态Q/K调整 |

---

## 7. 结论与展望

### 7.1 移植结论

✅ **技术层面**: 分头正交初始化成功移植到GPT
- 实现正确
- 训练稳定

⚠️ **性能层面**: 未达预期目标
- PPL损失9.8%，超出5%目标
- 生成任务对Q/K冻结敏感

### 7.2 下一步

进行Phase 3: 全面消融实验，探索：
- 不同数据集上的表现
- 正交初始化的必要性
- 可能的改进策略

---

**报告生成时间**: 2026-02-12
**实验完成时间**: 2026-02-09
