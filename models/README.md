# 模型目录 - Models

**整理时间**: 2026年2月6日
**整理者**: 张天禹 (Zhang Tianyu)
**学号**: s125mdg43_10

---

## 目录结构

### 01-预训练模型
训练完成的模型文件。

| 模型 | 大小 | 说明 |
|------|------|------|
| OELM_TinyStories_Small_v1.0.pt | 490MB | OELM Small模型，TinyStories数据集10K步训练 |

**模型信息**:
- 架构: Orthogonal ELM Transformer
- 配置: Small (41.7M参数)
- 数据集: TinyStories
- 训练步数: 10,000
- 验证Loss: 3.29
- 验证PPL: 26.87

### 02-检查点
训练过程中的检查点文件（由训练脚本自动生成）。

### 03-基准模型
Baseline模型（GPT等对比模型）。

### 04-导出模型
导出的ONNX/TensorRT等格式模型。

---

## 模型代码

| 文件 | 说明 |
|------|------|
| modeling_oelm.py | OELM模型实现 |
| modeling_gpt.py | GPT Baseline实现 |
| __init__.py | 模块导出 |

---

## 使用模型

### 加载预训练模型

```python
import torch
from models import create_oelm_small

# 创建模型
model = create_oelm_small()

# 加载权重
checkpoint = torch.load(
    '01-预训练模型/OELM_TinyStories_Small_v1.0.pt',
    map_location='cpu'
)
model.load_state_dict(checkpoint['model'])

# 查看信息
print(f"训练步数: {checkpoint['step']}")
print(f"最佳验证Loss: {checkpoint['best_val_loss']:.4f}")
```

### 模型推理

```python
model.eval()
input_ids = torch.randint(0, 1000, (1, 10))

with torch.no_grad():
    # 语言建模
    logits = model(input_ids, return_loss=False)
    # 或文本生成
    output = model.generate(input_ids, max_new_tokens=50)
```

---

## 模型对比

| 模型 | 参数 | 训练速度 | 显存占用 | 验证PPL |
|------|------|----------|----------|---------|
| OELM | 41.7M | 26,027 tok/s | 2.49 GB | 26.87 |
| GPT | 124.4M | 9,205 tok/s | 5.08 GB | - |

---

## 命名规范

- 格式: {模型名}_{数据集}_{配置}_v{版本}.pt
- 示例: OELM_TinyStories_Small_v1.0.pt

---

*整理完成于 2026年2月6日*
