# 训练日志 - Training Logs

**项目名称**: Orthogonal ELM Transformer
**作者**: 张天禹 (Zhang Tianyu)
**学号**: s125mdg43_10
**指导单位**: NTU MLDA Lab
**训练时间**: 2026年2月6日
**报告生成**: Claude Code AI Assistant

---

## 项目信息

- **项目名称**: Orthogonal ELM Transformer
- **作者**: 张天禹 (Zhang Tianyu)
- **学号**: s125mdg43_10
- **指导单位**: NTU MLDA Lab
- **训练时间**: 2026年2月6日
- **服务器**: NTU MLDA GPU Cluster (gpu43.dynip.ntu.edu.sg)

---

## 文件索引

### 核心报告
| 文件名 | 格式 | 大小 | 说明 |
|--------|------|------|------|
| OELM_TrainingReport_v1.0_20260206.pdf | PDF | 74KB | **完整训练报告（PDF版）** |
| TRAINING_REPORT.md | Markdown | 14KB | 完整训练报告（Markdown源文件） |

### 训练日志
| 文件名 | 格式 | 大小 | 说明 |
|--------|------|------|------|
| OELM_TrainingLog_Medium512_v1.0_20260206.md | Markdown | 12KB | **Medium-512配置对比实验日志** |

**日志内容**: GPT vs OELM在Medium-512配置下的详细训练记录，包含时间线、问题解决过程、性能对比和后续计划。

### 模型文件
| 文件名 | 格式 | 大小 | 说明 |
|--------|------|------|------|
| OELM_TinyStories_Small_v1.0.pt | PyTorch | 490MB | 验证集最佳模型（位于../models/01-预训练模型/） |

---

## 快速导航

### 报告内容
1. **模型架构** - OELM正交注意力机制详解
2. **训练配置** - 超参数、数据集、Baseline选择
3. **训练结果** - Loss曲线、性能对比
4. **测试结果** - OELM vs GPT效率分析
5. **结论展望** - 主要发现与未来工作

### 关键指标
| 指标 | 数值 |
|------|------|
| 训练速度 | 26,027 tokens/sec (2.83x vs GPT) |
| 内存占用 | 2.49 GB (51% reduction vs GPT) |
| 参数量 | 41.7M (66% reduction vs GPT) |
| 验证Loss | 3.29 (Perplexity: 26.87) |

---

## 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| v1.0 | 2026-02-06 | 初始训练报告，TinyStories数据集10K步训练 |
| v1.1 | 2026-02-06 | 添加Medium-512对比实验日志 (GPT vs OELM) |

---

## 联系方式
- **Email**: s125mdg43_10@ntu.edu.sg
- **服务器**: gpu43.dynip.ntu.edu.sg

---

*本报告由 Claude Code AI Assistant 协助生成*
*Generated with assistance from Claude Code AI Assistant*
