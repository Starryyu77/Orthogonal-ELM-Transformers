# Phase 6 实验执行指南

**创建时间**: 2026-03-11  
**状态**: 🟡 脚本已创建，等待上传到集群执行

---

## 📦 本地准备的文件

所有脚本已保存在:
```
Train/experiments/phase6-multidata/scripts/
├── finetune_multidata.py          # 主训练脚本 (支持4个数据集)
├── run_agnews_baseline.sh         # AG News Baseline
├── run_agnews_qk.sh               # AG News OELM-QK
├── run_agnews_qk_ffn.sh           # AG News OELM-QK-FFN
├── run_sst2_baseline.sh           # SST-2 Baseline
├── run_sst2_qk.sh                 # SST-2 OELM-QK
├── run_sst2_qk_ffn.sh             # SST-2 OELM-QK-FFN
├── run_xnli_baseline.sh           # XNLI Baseline
├── run_xnli_qk.sh                 # XNLI OELM-QK
├── run_xnli_qk_ffn.sh             # XNLI OELM-QK-FFN
├── run_mnli_baseline.sh           # MNLI Baseline
├── run_mnli_qk.sh                 # MNLI OELM-QK
├── run_mnli_qk_ffn.sh             # MNLI OELM-QK-FFN
└── submit_all_phase6.sh           # 一键提交所有任务
```

---

## 🚀 上传和执行步骤

### 方法1: 手动上传 (推荐)

```bash
# 1. 登录集群
ssh tianyu016@10.97.216.128

# 2. 进入项目目录
cd /projects/LlamaFactory/OELM-Pretrain

# 3. 创建目录
mkdir -p scripts logs outputs/phase6_multidata

# 4. 在本地Mac上打开新终端，复制文件
# (在本地执行)
scp /Users/starryyu/2026/Orthogonal\ ELM\ Transformers/Train/experiments/phase6-multidata/scripts/*.py tianyu016@10.97.216.128:/projects/LlamaFactory/OELM-Pretrain/scripts/
scp /Users/starryyu/2026/Orthogonal\ ELM\ Transformers/Train/experiments/phase6-multidata/scripts/*.sh tianyu016@10.97.216.128:/projects/LlamaFactory/OELM-Pretrain/scripts/

# 5. 回到集群终端，添加执行权限
chmod +x /projects/LlamaFactory/OELM-Pretrain/scripts/*.sh

# 6. 提交所有任务
./scripts/submit_all_phase6.sh
```

### 方法2: 逐个提交 (适合测试)

```bash
# 登录集群
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain

# AG News (先测试这个)
sbatch scripts/run_agnews_baseline.sh
sbatch scripts/run_agnews_qk.sh
sbatch scripts/run_agnews_qk_ffn.sh

# 查看作业状态
squeue -u tianyu016

# 其他数据集...
```

---

## 📊 实验执行追踪

| 数据集 | Baseline | OELM-QK | OELM-QK-FFN | 状态 |
|:-------|:--------:|:-------:|:-----------:|:----:|
| **AG News** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **SST-2** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **XNLI** | ☐ | ☐ | ☐ | 🟡 等待执行 |
| **MNLI** | ☐ | ☐ | ☐ | 🟡 等待执行 |

---

## 🔍 监控命令

```bash
# 查看作业队列
squeue -u tianyu016

# 查看特定作业日志
tail -f logs/agnews_baseline_12345.out

# 查看所有结果
cat outputs/phase6_multidata/ag_news/baseline/results.json
cat outputs/phase6_multidata/ag_news/oelm_qk/results.json
cat outputs/phase6_multidata/ag_news/oelm_qk_ffn/results.json

# 检查磁盘空间
df -h /projects/LlamaFactory/
```

---

## ⚠️ 注意事项

1. **GPU配额**: 一次只能申请1块Pro 6000，作业会排队执行
2. **预计时间**: 每个作业约1-3小时 (AG News ~1h, XNLI/MNLI ~3h)
3. **存储空间**: 12个作业约需要50GB存储空间
4. **依赖关系**: 所有作业独立，无需等待

---

## ✅ 完成标准

每个数据集完成时:
- [ ] 3个results.json文件生成
- [ ] 日志显示"Fine-tuning completed!"
- [ ] 最佳准确率 >= Baseline - 2%

所有数据集完成后:
- [ ] 运行结果汇总脚本
- [ ] 更新AI_HANDOFF.md
- [ ] 生成Phase 6报告

---

## 📞 问题排查

| 问题 | 解决方案 |
|:-----|:---------|
| ImportError | 检查conda环境是否激活 |
| CUDA OOM | 减小batch_size (16 -> 8) |
| 磁盘满 | 清理outputs/pretrain/中间检查点 |
| 作业pending | 等待其他作业完成，检查配额 |

---

**准备完成**: 2026-03-11  
**执行状态**: 等待上传到集群
