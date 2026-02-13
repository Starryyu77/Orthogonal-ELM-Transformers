# MLDA GPU 实验脚本

## 实验配置

- **模型尺寸**: n_layer=6, d_model=512, n_head=8, d_ff=2048
- **数据集**: TinyStories (或 WikiText-103)
- **序列长度**: 512
- **Batch Size**: 32 (每GPU)
- **训练步数**: 100000

## 脚本说明

| 脚本 | 用途 |
|------|------|
| `train_gpt_medium512.sh` | 训练标准 GPT 基线模型 |
| `train_oelm_medium512.sh` | 训练 OELM 模型 (freeze_ratio=0.075) |
| `train_oelm_freeze_experiments.sh` | 冻结比例对比实验 (0.25, 0.50) |
| `start_training_tmux.sh` | 使用 tmux 同时启动 GPT + OELM 训练 |

## 使用方法

### 方法1: 直接运行（当前终端）

```bash
# 训练 GPT
bash mlda_scripts/train_gpt_medium512.sh

# 训练 OELM（另一个终端）
bash mlda_scripts/train_oelm_medium512.sh
```

### 方法2: 使用 tmux（推荐，断开SSH继续运行）

```bash
# 同时启动 GPT + OELM 训练
bash mlda_scripts/start_training_tmux.sh

# 查看训练状态
tmux attach -t oelm_gpt    # 查看GPT
tmux attach -t oelm_oelm   # 查看OELM
```

### 方法3: 后台运行

```bash
# GPT 后台运行
nohup bash mlda_scripts/train_gpt_medium512.sh > logs/gpt_train.log 2>&1 &

# OELM 后台运行
nohup bash mlda_scripts/train_oelm_medium512.sh > logs/oelm_train.log 2>&1 &

# 查看运行状态
tail -f logs/gpt_train.log
tail -f logs/oelm_train.log
```

## 监控训练

1. **WandB**: 登录 https://wandb.ai 查看实时训练曲线
2. **日志文件**: `models/checkpoints/*/logs/`
3. **Checkpoint**: 每 5000 步自动保存到 `models/checkpoints/`

## 对比指标

训练完成后比较：

| 指标 | GPT | OELM |
|------|-----|------|
| Validation Loss | ? | ? |
| Training Speed (tokens/sec) | ? | ? |
| 参数量 | ~44M | ~30M |
| 训练时间 | ? | ? |

## 注意事项

1. 确保有 `oelm` conda 环境
2. 确保数据已准备好：`data/tiny_stories/train.bin`
3. 使用 `nvidia-smi` 监控 GPU 使用率
4. 2小时限制：使用 tmux 或 nohup 保持训练
