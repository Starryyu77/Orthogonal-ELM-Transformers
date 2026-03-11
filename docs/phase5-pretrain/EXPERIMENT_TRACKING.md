# OELM Pretrain 实验跟踪

## 当前状态

| 作业ID | 方法 | 状态 | 运行时间 | 备注 |
|--------|------|------|----------|------|
| 43704 | Baseline | Running | ~3分钟 | Tokenizing数据中 |
| - | OELM-QK | Pending | - | 等待Baseline完成 |
| - | OELM-QK-FFN | Pending | - | 等待OELM-QK完成 |

## 实验队列

### Phase 1: 预训练
```
[进行中] Baseline (TinyStories, 1 epoch)
[队列]   OELM-QK
[队列]   OELM-QK-FFN
```

### Phase 2: 下游微调
```
[待开始] Baseline + IMDB
[待开始] OELM-QK + IMDB
[待开始] OELM-QK-FFN + IMDB
```

## 监控命令

```bash
# SSH到集群
ssh ntu-cluster

# 查看作业状态
cd /projects/LlamaFactory/OELM-Pretrain
./scripts/monitor.sh

# 或手动查看
watch squeue -u tianyu016
tail -f logs/pretrain_baseline-43704.out
```

## 预期时间线

| 阶段 | 预计时间 | 状态 |
|------|----------|------|
| Baseline预训练 | 4-8小时 | 进行中 |
| OELM-QK预训练 | 4-8小时 | 待开始 |
| OELM-QK-FFN预训练 | 4-8小时 | 待开始 |
| 下游微调(3个) | 1-2小时 | 待开始 |
| **总计** | **约24小时** | - |

## 关键指标

### 预训练成功标准
- PPL (Perplexity) 差距 < 10%
- 训练稳定，无NaN/Inf

### 下游微调成功标准
- 准确率与Baseline相当或更好
- 参数效率（可训练参数减少）

## 检查点位置

```
/projects/LlamaFactory/OELM-Pretrain/outputs/pretrain/
├── baseline/
│   └── checkpoint-*/
├── oelm_qk/
│   └── checkpoint-*/
└── oelm_qk_ffn/
    └── checkpoint-*/
```

## 下一步操作

1. **等待Baseline完成** (当前)
2. **提交OELM-QK作业**
   ```bash
   sbatch scripts/run_pretrain_oelm_qk.sh
   ```
3. **提交OELM-QK-FFN作业**
   ```bash
   sbatch scripts/run_pretrain_oelm_qk_ffn.sh
   ```
4. **运行下游微调**
   ```bash
   sbatch scripts/run_finetune.sh \
       outputs/pretrain/baseline/best/pytorch_model.pt \
       imdb baseline
   ```

## 注意事项

- 每个作业使用1个Pro 6000 GPU
- 最多同时运行2个作业（根据配额）
- 数据缓存位置: `/projects/LlamaFactory/.cache/huggingface`

---
最后更新: 2026-03-09 07:40 UTC