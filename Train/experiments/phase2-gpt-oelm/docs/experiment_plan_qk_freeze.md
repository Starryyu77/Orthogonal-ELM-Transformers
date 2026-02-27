# Q/K矩阵冻结机制实验设计方案

## 实验目标

验证ELM理论的核心假设：**冻结随机正交投影矩阵(Q/K)可以在减少可训练参数的同时保持或提升模型性能**。

## 背景分析

### 当前状态

根据代码分析，`OrthogonalLinear`类已实现冻结机制（使用`register_buffer`），但之前的训练日志显示：

```
OELM Medium-512 | 41.8M (93%) | 41.8M (100%) | 0
注: 当前OELM实现未启用冻结机制
```

这表明存在**实现与训练的断层**。

### 问题诊断

可能的原因：
1. **模型实例化问题** - 训练时使用了错误的模型配置
2. **参数覆盖** - 检查点加载时覆盖了冻结参数
3. **统计信息错误** - 参数统计方法未正确识别buffer
4. **实现Bug** - `OrthogonalLinear`类存在缺陷

---

## 实验架构

### 实验组设计 (3组)

| 实验组 | 模型 | Q/K状态 | 可训练参数 | 预期效果 |
|--------|------|---------|------------|----------|
| **Group A** | GPT-Base | 无正交初始化 | 44.9M (100%) | 基线 |
| **Group B** | OELM-NoFreeze | 正交初始化, 不冻结 | 41.8M (93%) | 验证正交初始化效果 |
| **Group C** | OELM-Freeze | 正交初始化, 冻结Q/K | ~38M (85%) | 验证ELM理论 |

### 关键假设验证

```
H1: OELM-Freeze的可训练参数比OELM-NoFreeze少 ~15%
H2: OELM-Freeze的Val PPL ≈ OELM-NoFreeze (性能保持)
H3: OELM-Freeze的训练速度 > OELM-NoFreeze (效率提升)
H4: OELM-Freeze的最终性能 ≥ GPT-Base (竞争力验证)
```

---

## Phase 1: 诊断与修复 (1-2天)

### 1.1 诊断当前实现

**目标**: 确定为什么之前的训练显示"未启用冻结机制"

**任务清单**:
- [ ] 创建诊断脚本 `scripts/diagnose_freeze.py`
- [ ] 打印所有参数及其requires_grad状态
- [ ] 验证OrthogonalLinear的weight是否为buffer
- [ ] 检查训练脚本中的模型实例化
- [ ] 模拟训练步骤，观察参数是否变化

**诊断代码框架**:
```python
# scripts/diagnose_freeze.py
from models.modeling_oelm import OrthogonalELMTransformer

model = OrthogonalELMTransformer(...)

# 打印参数统计
for name, param in model.named_parameters():
    print(f"Parameter: {name}, shape: {param.shape}, requires_grad: {param.requires_grad}")

# 打印buffer
for name, buffer in model.named_buffers():
    print(f"Buffer: {name}, shape: {buffer.shape}")

# 验证统计信息
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen = total - trainable
print(f"Total: {total}, Trainable: {trainable}, Frozen: {frozen}")
```

### 1.2 修复实现 (如需要)

如果发现bug，修复优先级：

1. **High**: OrthogonalLinear buffer注册
2. **Medium**: 模型统计信息计算
3. **Low**: 检查点加载逻辑

---

## Phase 2: 实现验证 (2-3天)

### 2.1 创建三组实验代码

**Group A: GPT-Base** (已存在)
```bash
# 使用现有实现
python train.py --model_type gpt ...
```

**Group B: OELM-NoFreeze** (修改现有)
```python
# models/modeling_oelm.py 添加配置选项
class OrthogonalMultiHeadAttention(nn.Module):
    def __init__(self, ..., freeze_qk=True):  # 添加参数
        ...
        if freeze_qk:
            # 使用buffer (冻结)
            self.register_buffer('weight', weight)
        else:
            # 使用Parameter (可训练)
            self.weight = nn.Parameter(weight)
```

**Group C: OELM-Freeze** (默认实现)
```python
# 确保freeze_qk=True为默认值
```

### 2.2 创建统一的实验脚本

**文件**: `scripts/experiment_qk_freeze.py`

```python
"""
Q/K冻结机制对比实验
运行三组实验并自动收集结果
"""

EXPERIMENTS = {
    'gpt_base': {
        'model_type': 'gpt',
        'freeze_qk': None,
        'gpus': '0,1',
        'port': 29500,
    },
    'oelm_no_freeze': {
        'model_type': 'oelm',
        'freeze_qk': False,
        'gpus': '2,3',
        'port': 29501,
    },
    'oelm_freeze': {
        'model_type': 'oelm',
        'freeze_qk': True,
        'gpus': '2,3',
        'port': 29502,
    }
}
```

---

## Phase 3: 对比实验 (3-4天)

### 3.1 实验配置

**固定参数** (所有组相同):
```yaml
n_layers: 6
d_model: 512
n_heads: 8
d_ff: 2048
seq_len: 512
batch_size: 8
max_steps: 100000
learning_rate: 3e-4 (warmup 2K steps, cosine decay)
dataset: TinyStories
```

**硬件分配**:
- Group A (GPT): GPU 0,1
- Group B (OELM-NoFreeze): GPU 2,3
- Group C (OELM-Freeze): GPU 2,3 (顺序运行或与B并行)

### 3.2 训练计划

**方案1: 顺序运行** (资源充足)
```
Day 1: Group A (GPT) + Group B (OELM-NoFreeze) 同时
Day 2-3: Group C (OELM-Freeze)
```

**方案2: 并行运行** (如果GPU足够)
```
Day 1-2: 三组同时运行
- GPT: GPU 0,1
- OELM-NoFreeze: GPU 2
- OELM-Freeze: GPU 3
```

---

## Phase 4: 数据收集与分析 (1天)

### 4.1 自动收集指标

**创建分析脚本**: `scripts/analyze_freeze_experiment.py`

**收集指标**:
1. **参数统计**
   - 总参数量
   - 可训练参数量
   - 冻结参数量
   - 各层参数分布

2. **训练指标**
   - 训练Loss曲线
   - 验证PPL曲线
   - 学习率变化
   - 训练速度 (steps/小时)

3. **效率指标**
   - 达到目标PPL所需时间
   - 显存占用
   - 收敛速度

### 4.2 可视化

**生成图表**:
1. 三组Val PPL对比曲线
2. 参数效率对比 (PPL/参数)
3. 训练速度对比
4. 收敛时间对比

---

## Phase 5: 结果验证 (1天)

### 5.1 假设验证

| 假设 | 验证方法 | 通过标准 |
|------|----------|----------|
| H1: 参数减少15% | 对比Group B和C的参数统计 | C的冻结参数 > 3M |
| H2: 性能保持 | 对比最终Val PPL | \|PPL_C - PPL_B\| < 5% |
| H3: 速度提升 | 对比训练速度 | Speed_C > Speed_B |
| H4: 竞争力 | 对比Group A | PPL_C ≤ PPL_A + 10% |

### 5.2 统计显著性

- 每组至少运行1次完整100K steps
- (可选) 每组运行3次不同随机种子，取平均

---

## 实施时间表

| 阶段 | 任务 | 预计时间 | 依赖 |
|------|------|----------|------|
| **Day 1** | Phase 1.1: 诊断 | 4-6h | 无 |
| **Day 1-2** | Phase 1.2: 修复 | 2-4h | 1.1 |
| **Day 2** | Phase 2: 代码实现 | 6-8h | 1.2 |
| **Day 3-4** | Phase 3: 训练 | 12-16h | 2 |
| **Day 5** | Phase 4-5: 分析 | 4-6h | 3 |
| **总计** | | **5天** | |

---

## 代码修改清单

### 修改文件1: `models/modeling_oelm.py`

```python
# 修改OrthogonalLinear类，支持可配置冻结
class OrthogonalLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=False,
                 ortho_method='qr', freeze=True):  # 添加freeze参数
        super().__init__()
        self.freeze = freeze

        # 正交初始化
        weight = self._orthogonal_init(out_features, in_features, ortho_method)

        if freeze:
            self.register_buffer('weight', weight)
        else:
            self.weight = nn.Parameter(weight)

        # bias处理...

# 修改OrthogonalMultiHeadAttention，传递freeze参数
class OrthogonalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1,
                 ortho_method='qr', freeze_qk=True):  # 添加freeze_qk
        ...
        self.W_q = OrthogonalLinear(..., freeze=freeze_qk)
        self.W_k = OrthogonalLinear(..., freeze=freeze_qk)
```

### 修改文件2: `models/modeling_oelm.py` (Transformer层)

```python
# 修改TransformerBlock，传递freeze_qk
class OELMTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1,
                 ortho_method='qr', freeze_qk=True):
        ...
        self.attention = OrthogonalMultiHeadAttention(
            ..., freeze_qk=freeze_qk)

# 修改OrthogonalELMTransformer，添加freeze_qk配置
class OrthogonalELMTransformer(PreTrainedModel):
    def __init__(self, config):
        ...
        freeze_qk = getattr(config, 'freeze_qk', True)  # 默认True

        self.blocks = nn.ModuleList([
            OELMTransformerBlock(..., freeze_qk=freeze_qk)
            for _ in range(config.num_layers)
        ])
```

### 修改文件3: `scripts/02-训练脚本/train.py`

```python
# 添加命令行参数
parser.add_argument('--freeze_qk', type=lambda x: x.lower() == 'true',
                    default=True, help='Freeze Q/K matrices (OELM only)')

# 模型实例化时传递参数
if args.model_type == 'oelm':
    config.freeze_qk = args.freeze_qk
```

### 新增文件: `scripts/experiment_qk_freeze.py`

统一实验控制脚本，同时管理三组实验。

---

## 预期成果

### 1. 技术成果

- [ ] 修复/验证OELM冻结机制
- [ ] 量化冻结Q/K对性能和效率的影响
- [ ] 验证ELM理论在Transformer上的适用性

### 2. 文档成果

- [ ] 实验报告 (markdown + PDF)
- [ ] 修改后的代码 (PR ready)
- [ ] 可视化结果 (图表)
- [ ] 结论与建议

### 3. 潜在发现

| 情景 | 结论 | 后续行动 |
|------|------|----------|
| Freeze效果 > NoFreeze | ELM理论验证成功 | 推广到更大模型 |
| Freeze效果 ≈ NoFreeze | ELM有一定效果 | 优化实现 |
| Freeze效果 < NoFreeze | 需要重新思考 | 分析原因 |

---

## 风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 诊断发现重大Bug | 中 | 高 | 预留修复时间 |
| 训练中断 | 低 | 中 | 定期保存检查点 |
| GPU资源不足 | 中 | 中 | 顺序运行方案 |
| 结果不显著 | 中 | 中 | 多次运行取平均 |
| 超出时间预算 | 低 | 低 | 分阶段交付 |

---

## 下一步行动

1. **立即执行**: 创建并运行诊断脚本
2. **根据诊断结果**: 决定是否需要修复
3. **代码修改**: 实现freeze_qk配置选项
4. **启动实验**: 三组对比训练
5. **分析结果**: 验证ELM理论

---

**文档版本**: v1.0
**创建时间**: 2026-02-07
**作者**: Claude Code AI Assistant
