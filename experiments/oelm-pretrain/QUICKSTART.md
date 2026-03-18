# OELM 预训练实验 - 快速开始指南

**准备就绪！** 现在可以开始执行阶段1实验了。

---

## 🚀 立即开始

### 1. 登录服务器

```bash
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain
```

### 2. 创建日志目录

```bash
mkdir -p logs
```

### 3. 启动阶段1实验（3组并行）

```bash
# 启动 Baseline
cd /projects/LlamaFactory/OELM-Pretrain
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh baseline

# 启动 OELM-QK
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh oelm_qk

# 启动 OELM-QK-FFN
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh oelm_qk_ffn
```

---

## 📊 实验配置（阶段1）

| 配置项 | 值 |
|:-------|:---|
| **模型** | GPT-2 Mini (42M params) |
| **架构** | d_model=512, n_layer=6, n_head=8 |
| **数据** | TinyStories (2GB) |
| **训练步数** | 10,000 steps |
| **批次大小** | 32 |
| **序列长度** | 512 |
| **学习率** | 5e-4 |
| **预计时间** | 3-4小时 |
| **GPU** | 1× Pro 6000 |

---

## 🔍 监控实验

### 查看运行状态

```bash
# 查看所有作业
squeue -u tianyu016

# 查看输出日志（实时）
tail -f logs/stage1_mini_*.out

# 查看GPU使用率
watch -n 1 nvidia-smi
```

### 检查训练进度

```bash
# 查看最新的日志输出
tail -100 logs/stage1_mini_*.out | grep "Step"

# 查看保存的检查点
ls -lh outputs/stage1_tinystories_mini/*/checkpoint-*
```

---

## 📈 预期结果

### 成功标准

| 指标 | 通过标准 |
|:-----|:---------|
| Loss下降 | 所有组持续下降 |
| PPL差距 | < 5% (vs Baseline) |
| 训练速度 | 差异 < 20% |
| 下游准确率 | > 随机水平 |

### 理想结果

```
Baseline:    PPL = 15-20
OELM-QK:     PPL = 15-21 (差距 < 5%)
OELM-QK-FFN: PPL = 16-22 (差距 < 10%)
```

---

## 📝 实验完成后

### 1. 评估下游任务（阶段2）

```bash
# 运行快速评估
python experiments/oelm-pretrain/scripts/evaluate_quick.py \
    --checkpoint outputs/stage1_tinystories_mini/baseline/final \
    --dataset sst2 \
    --output_dir results/stage1/baseline_sst2
```

### 2. 更新进度文档

```bash
# 编辑 EXPERIMENT_PROGRESS.md
vim EXPERIMENT_PROGRESS.md

# 更新阶段1的结果
git add EXPERIMENT_PROGRESS.md
git commit -m "update: Stage 1 results"
git push origin main
```

### 3. 决策点

根据结果决定：
- ✅ **通过** → 进入阶段3 (OpenWebText)
- ⚠️ **勉强** → 调整参数重试
- ❌ **失败** → 分析原因，修改策略

---

## 📂 输出文件

实验完成后，检查以下文件：

```
outputs/stage1_tinystories_mini/
├── baseline/
│   ├── checkpoint-2000/
│   ├── checkpoint-4000/
│   ├── checkpoint-6000/
│   ├── checkpoint-8000/
│   └── final/
│       ├── pytorch_model.bin
│       ├── config.json
│       └── model_config.json
├── oelm_qk/
│   └── ...
└── oelm_qk_ffn/
    └── ...
```

---

## 🆘 故障排除

### 如果作业失败

```bash
# 查看错误日志
cat logs/stage1_mini_*.err

# 检查Python错误
tail -50 logs/stage1_mini_*.out

# 重新提交
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh baseline
```

### 常见错误

1. **CUDA out of memory**
   - 减小 `--batch_size` 到 16
   
2. **数据集下载失败**
   - 检查网络连接
   - 手动下载：`python -c "from datasets import load_dataset; load_dataset('roneneldan/TinyStories')"`

3. **权限错误**
   - 检查脚本权限：`chmod +x experiments/oelm-pretrain/scripts/*.sh`

---

## 📞 联系

遇到问题随时联系！

- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers
- **服务器**: NTU EEE GPU Cluster

---

**开始实验吧！** 🚀

每完成一组实验，记得更新 `EXPERIMENT_PROGRESS.md`！
