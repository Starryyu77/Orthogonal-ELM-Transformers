# Phase 3 完成报告: 实验控制脚本

## 完成时间
2026-02-07

## 新增文件

| 文件 | 用途 |
|------|------|
| `scripts/experiment_qk_freeze.py` | 统一实验控制脚本，管理三组实验 |
| `scripts/analyze_freeze_experiment.py` | 实验结果分析脚本，生成对比报告 |

## 功能说明

### 1. experiment_qk_freeze.py

统一实验控制脚本，支持以下功能：

#### 运行模式
```bash
# 顺序运行 (推荐，节约GPU资源)
python scripts/experiment_qk_freeze.py --mode sequential

# 并行运行 (需要足够GPU)
python scripts/experiment_qk_freeze.py --mode parallel

# 只运行特定组
python scripts/experiment_qk_freeze.py --groups gpt_base,oelm_freeze
```

#### 远程服务器支持
```bash
# 在MLDA服务器上运行
python scripts/experiment_qk_freeze.py --remote --host gpu43.dynip.ntu.edu.sg --user s125mdg43_10
```

#### 干运行模式
```bash
# 测试配置，不实际执行
python scripts/experiment_qk_freeze.py --dry-run
```

#### 实验配置

脚本内置三组实验配置：

| 实验ID | 名称 | 模型类型 | freeze_qk | GPU | 端口 |
|--------|------|----------|-----------|-----|------|
| gpt_base | Group A: GPT-Base | gpt | - | 0,1 | 29500 |
| oelm_no_freeze | Group B: OELM-NoFreeze | oelm | false | 2,3 | 29501 |
| oelm_freeze | Group C: OELM-Freeze | oelm | true | 0,1,2,3 | 29502 |

### 2. analyze_freeze_experiment.py

实验结果分析脚本，功能包括：

- 解析训练日志，提取Loss和PPL
- 统计模型参数 (总参数/可训练参数/冻结参数)
- 生成对比表格
- 验证四个核心假设 (H1-H4)
- 生成可视化图表

#### 使用方法
```bash
# 基本分析
python scripts/analyze_freeze_experiment.py

# 生成图表
python scripts/analyze_freeze_experiment.py --plot

# 指定输出目录
python scripts/analyze_freeze_experiment.py --output experiments/analysis
```

## 测试验证

### 干运行测试
```bash
$ python scripts/experiment_qk_freeze.py --dry-run

======================================================================
Q/K矩阵冻结机制对比实验
======================================================================
实验组: gpt_base, oelm_no_freeze, oelm_freeze
运行模式: sequential
======================================================================

顺序运行模式
======================================================================

启动实验: Group A: GPT-Base
命令: python -m torch.distributed.run --nproc_per_node 2 --master_port 29500 ...
GPU: 0,1
输出目录: models/checkpoints/exp_gpt_base
[DRY RUN] 不实际执行训练

启动实验: Group B: OELM-NoFreeze
命令: python -m torch.distributed.run --nproc_per_node 2 --master_port 29501 ... --freeze_qk false ...
GPU: 2,3
输出目录: models/checkpoints/exp_oelm_no_freeze
[DRY RUN] 不实际执行训练

启动实验: Group C: OELM-Freeze
命令: python -m torch.distributed.run --nproc_per_node 4 --master_port 29502 ... --freeze_qk true ...
GPU: 0,1,2,3
输出目录: models/checkpoints/exp_oelm_freeze
[DRY RUN] 不实际执行训练
```

所有三组实验配置正确生成！

## 输出文件结构

实验完成后，将生成以下文件结构：

```
models/checkpoints/
├── exp_gpt_base/
│   ├── training.log          # 训练日志
│   ├── experiment_result.json # 实验结果
│   ├── best_model.pt         # 最佳模型
│   └── checkpoint_*.pt       # 检查点
├── exp_oelm_no_freeze/
│   ├── training.log
│   ├── experiment_result.json
│   ├── best_model.pt
│   └── checkpoint_*.pt
└── exp_oelm_freeze/
    ├── training.log
    ├── experiment_result.json
    ├── best_model.pt
    └── checkpoint_*.pt

experiments/
├── qk_freeze_summary.json    # 实验总结
└── analysis/                  # 分析结果
    ├── comparison_table.txt   # 对比表格
    ├── hypothesis_validation.txt # 假设验证
    ├── analysis_results.json  # 完整结果
    ├── val_ppl_comparison.png # PPL对比图
    └── parameter_distribution.png # 参数分布图
```

## 下一步: Phase 4 实验执行

### 本地小规模验证
```bash
# 测试训练脚本的新参数
python scripts/02-训练脚本/train.py \
    --model_type oelm --freeze_qk true \
    --d_model 256 --num_layers 2 --num_heads 4 \
    --max_steps 100 --batch_size 4 \
    --data_path data/tiny_stories/train.bin
```

### MLDA服务器实验
```bash
# 1. 同步代码到服务器
./mlda-run.sh sync

# 2. 在服务器上运行实验
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
cd ~/Orthogonal_ELM_Transformers/Train

# 3. 启动顺序实验
python scripts/experiment_qk_freeze.py --mode sequential

# 或分别启动
# Group A (GPT-Base)
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29500 \
    scripts/02-训练脚本/train.py \
    --model_type gpt \
    --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 8 --max_steps 100000 \
    --data_path data/tiny_stories/train.bin \
    --out_dir models/checkpoints/exp_gpt_base

# Group B (OELM-NoFreeze)
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29501 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --freeze_qk false \
    --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 8 --max_steps 100000 \
    --data_path data/tiny_stories/train.bin \
    --out_dir models/checkpoints/exp_oelm_no_freeze

# Group C (OELM-Freeze)
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=29502 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --freeze_qk true \
    --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 8 --max_steps 100000 \
    --data_path data/tiny_stories/train.bin \
    --out_dir models/checkpoints/exp_oelm_freeze
```

### 实验计划

| 时间 | 实验 | GPU | 预计时长 |
|------|------|-----|----------|
| Day 1 | Group A (GPT-Base) | 0,1 | 10-12h |
| Day 1 | Group B (OELM-NoFreeze) | 2,3 | 10-12h |
| Day 2 | Group C (OELM-Freeze) | 0,1,2,3 | 8-10h |

### 监控实验

```bash
# 查看训练状态
./mlda-run.sh status

# 查看实时日志
tail -f models/checkpoints/exp_gpt_base/training.log
tail -f models/checkpoints/exp_oelm_no_freeze/training.log
tail -f models/checkpoints/exp_oelm_freeze/training.log
```

## 注意事项

1. **确保数据文件存在**: 训练前确认 `data/tiny_stories/train.bin` 和 `val.bin` 存在
2. **GPU内存**: 每组实验使用2-4个GPU，确保显存充足
3. **检查点保存**: 每5000步自动保存检查点，防止训练中断
4. **日志记录**: 所有输出同时保存到文件和打印到控制台

## 实验完成后

```bash
# 1. 下载结果到本地
rsync -avz s125mdg43_10@gpu43.dynip.ntu.edu.sg:~/Orthogonal_ELM_Transformers/Train/models/checkpoints/ ./models/checkpoints/

# 2. 运行分析脚本
python scripts/analyze_freeze_experiment.py --plot

# 3. 查看分析报告
cat experiments/analysis/comparison_table.txt
cat experiments/analysis/hypothesis_validation.txt
```

## 总结

Phase 3完成，现在具备完整的实验执行和分析能力：

- ✅ 统一实验控制脚本
- ✅ 顺序/并行运行模式
- ✅ 远程服务器支持
- ✅ 自动化结果分析
- ✅ 假设验证和可视化

**准备进入Phase 4: 实验执行！**
