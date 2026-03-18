# Orthogonal ELM Transformers - 项目状态记录

**最后更新**: 2025年3月19日  
**当前状态**: ✅ 实验准备就绪，等待执行阶段1

---

## 🎉 完成的工作

### 1. 仓库完全重构

**旧状态**: Train/ 目录混乱，包含 255+ 个混乱文件  
**新状态**: 清晰的 experiments/ 结构，6 个独立实验

**执行的清理**:
- ✅ 删除 Train/ 目录（255 个文件，-88,580 行）
- ✅ 创建新的 README.md（清晰的实验导航）
- ✅ 创建 experiments/ 目录结构
- ✅ 移动核心模型代码到 shared/models/
- ✅ GitHub 提交: `b121aa3`

### 2. 预训练实验计划

**创建文件**:
- ✅ `EXPERIMENT_PLAN.md` - 完整5阶段实验计划
- ✅ `EXPERIMENT_PROGRESS.md` - 实时进度跟踪
- ✅ `QUICKSTART.md` - 快速开始指南
- ✅ `pretrain_mini.py` - Mini模型训练脚本（42M参数）
- ✅ `run_stage1_mini.sh` - 阶段1执行脚本

---

## 📁 当前目录结构

```
Orthogonal-ELM-Transformers/
├── README.md                    # 项目总览和实验导航
├── EXPERIMENT_PLAN.md           # ⭐ 完整实验计划（5阶段）
├── EXPERIMENT_PROGRESS.md       # ⭐ 实时进度跟踪
├── LICENSE                      # MIT 许可证
│
├── experiments/                 # 实验目录
│   ├── exp01-06/               # 已完成的历史实验
│   ├── shared/models/          # 共享模型代码
│   └── oelm-pretrain/          # ⭐ 新预训练实验
│       ├── QUICKSTART.md       # ⭐ 快速开始指南
│       └── scripts/
│           ├── pretrain_mini.py      # ⭐ Mini模型训练
│           ├── run_stage1_mini.sh    # ⭐ 阶段1执行
│           └── run_stage3_openwebtext.sh
│
└── .git/
```

---

## 🚀 准备执行的实验

### 阶段1: TinyStories 快速验证（Mini模型）

**配置**:
| 参数 | 值 |
|:-----|:---|
| 模型 | GPT-2 Mini (42M params) |
| 架构 | d=512, l=6, h=8 |
| 数据 | TinyStories (2GB) |
| 训练 | 10,000 steps |
| GPU | 1× Pro 6000 |
| 预计时间 | 3-4小时 |

**实验组**:
1. **Baseline** - 全可训练
2. **OELM-QK** - 冻结Q/K投影
3. **OELM-QK-FFN** - 冻结Q/K + FFN

**执行命令**:
```bash
# 登录服务器
ssh tianyu016@10.97.216.128
cd /projects/LlamaFactory/OELM-Pretrain

# 启动3组实验
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh baseline
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh oelm_qk
sbatch experiments/oelm-pretrain/scripts/run_stage1_mini.sh oelm_qk_ffn
```

---

## 📝 下一步行动

### 立即执行
- [ ] 登录 NTU 服务器
- [ ] 启动阶段1三组实验
- [ ] 监控训练进度
- [ ] 记录训练结果

### 阶段1完成后
- [ ] 更新 EXPERIMENT_PROGRESS.md
- [ ] 执行下游快速评估
- [ ] 根据结果决定阶段3

---

## 🔗 相关链接

- **GitHub**: https://github.com/Starryyu77/Orthogonal-ELM-Transformers
- **最新提交**: `a3d45fc` - feat: Add mini model for Stage 1
- **实验计划**: `EXPERIMENT_PLAN.md`
- **快速开始**: `experiments/oelm-pretrain/QUICKSTART.md`
- **服务器**: NTU EEE GPU Cluster (10.97.216.128)

---

## ✅ 待办清单

- [x] 分析现有实验结构
- [x] 创建新的根目录 README.md
- [x] 创建 experiments/ 目录结构
- [x] 整理 Phase 1-6 到新目录
- [x] 删除旧的 Train/ 目录
- [x] 创建完整实验计划 (EXPERIMENT_PLAN.md)
- [x] 创建进度跟踪文档 (EXPERIMENT_PROGRESS.md)
- [x] 创建 Mini 模型训练脚本
- [x] 创建阶段1执行脚本
- [x] 创建快速开始指南
- [x] 提交所有更改到 GitHub
- [ ] ⏳ 执行阶段1实验
- [ ] ⏳ 更新实验进度

---

**状态**: 🟢 准备就绪，等待开始实验！
