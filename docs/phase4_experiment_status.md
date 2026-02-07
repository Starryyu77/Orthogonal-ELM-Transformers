# Phase 4 实验执行状态报告

## 实验启动时间
2026-02-07 01:36

## 当前状态: ✅ 运行中

### 已启动实验

| 实验组 | 模型 | 状态 | GPU | 启动时间 |
|--------|------|------|-----|----------|
| **Group A** | GPT-Base | 🟢 运行中 | 0,1 | 01:35 |
| **Group B** | OELM-NoFreeze | 🟢 运行中 | 2,3 | 01:35 |
| **Group C** | OELM-Freeze | ⏳ 待启动 | - | - |

### GPU状态

```
GPU 0: 100% 利用率 | 196W / 200W | 82°C | Group A
GPU 1: 100% 利用率 | 195W / 200W | 80°C | Group A
GPU 2: 100% 利用率 | 197W / 200W | 79°C | Group B
GPU 3: 100% 利用率 | 197W / 200W | 80°C | Group B
```

### 训练进度

#### Group A (GPT-Base)
```
总参数: 44,896,768 (100% 可训练)
当前进度: Step ~100+
初始Loss: 10.93
初始PPL: 22026
状态: 正常收敛
```

#### Group B (OELM-NoFreeze)
```
总参数: 44,896,768 (100% 可训练)
Q/K frozen: False
当前进度: Step ~100+
初始Loss: 10.91
初始PPL: 22026
状态: 正常收敛
```

#### Group C (OELM-Freeze) - 待启动
```
预计参数: 44,896,768 (93% 可训练, 7% 冻结)
Q/K frozen: True
预计节省参数: ~3.1M
启动条件: Group A/B 完成后
```

## 监控命令

### 查看训练日志
```bash
# Group A
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'tail -f ~/Orthogonal_ELM_Transformers/Train/models/checkpoints/exp_gpt_base/training.log'

# Group B
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'tail -f ~/Orthogonal_ELM_Transformers/Train/models/checkpoints/exp_oelm_no_freeze/training.log'

# Group C (启动后)
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'tail -f ~/Orthogonal_ELM_Transformers/Train/models/checkpoints/exp_oelm_freeze/training.log'
```

### 查看GPU状态
```bash
./mlda-run.sh status
```

### 查看screen会话
```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'screen -ls'
```

## 启动Group C

当Group A和B完成后，在服务器上运行:

```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
cd ~/Orthogonal_ELM_Transformers/Train
chmod +x scripts/start_exp_c.sh
./scripts/start_exp_c.sh
```

或在本地运行:
```bash
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'bash ~/Orthogonal_ELM_Transformers/Train/scripts/start_exp_c.sh'
```

## 实验完成检查

实验完成时，每个组会生成:

```
models/checkpoints/exp_*/
├── training.log          # 完整训练日志
├── best_model.pt         # 最佳模型检查点
├── final.pt              # 最终模型检查点
├── checkpoint_*.pt       # 中间检查点
└── config.json           # 实验配置
```

## 预计完成时间

| 实验组 | 预计时长 | 预计完成 |
|--------|----------|----------|
| Group A | 10-12小时 | 2月7日 12:00 |
| Group B | 10-12小时 | 2月7日 12:00 |
| Group C | 8-10小时 | 2月7日 22:00 |

## 下一步操作

1. **监控训练**: 定期使用上述命令检查进度
2. **启动Group C**: 当A/B完成后启动
3. **结果分析**: 所有实验完成后运行分析脚本

## 故障排除

### 如果实验中断
```bash
# 重新连接screen会话
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
screen -r exp_gpt    # Group A
screen -r exp_oelm_nf # Group B
screen -r exp_oelm_f  # Group C
```

### 如果需要重启实验
```bash
# 停止现有进程
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg 'pkill -f "train.py"'

# 重新启动 (使用resume参数)
python train.py --resume models/checkpoints/exp_*/checkpoint_*.pt ...
```

## 联系信息

- **服务器**: gpu43.dynip.ntu.edu.sg
- **用户名**: s125mdg43_10
- **项目目录**: ~/Orthogonal_ELM_Transformers/Train
