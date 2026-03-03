# NTU GPU43 服务器运行指南

## 服务器信息

- **主机**: `10.97.216.128` (gpu43)
- **用户名**: `tianyu016`
- **项目路径**: `/projects/Orthogonal_ELM_Transformers/Train`
- **GPU**: 4x RTX A5000

## SSH连接

```bash
ssh tianyu016@10.97.216.128
```

## 同步代码到服务器

### 方法1: 使用rsync（推荐）

在本地机器上运行：

```bash
cd /path/to/Orthogonal_ELM_Transformers

rsync -avz --progress \
    --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
    --exclude 'venv' --exclude '.claude' --exclude 'out' \
    --exclude '*.pt' --exclude '*.bin' --exclude '.DS_Store' \
    --exclude 'data/*.bin' --exclude 'logs' \
    "./" "tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/"
```

### 方法2: 使用SCP

```bash
scp -r Train/experiments/phase4-gpt-classification/models/*.py \
    tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/Train/experiments/phase4-gpt-classification/models/

scp -r Train/experiments/phase4-gpt-classification/scripts/*.sh \
    tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/Train/experiments/phase4-gpt-classification/scripts/
```

## 运行实验

### 步骤1: 登录服务器

```bash
ssh tianyu016@10.97.216.128
cd /projects/Orthogonal_ELM_Transformers/Train/experiments/phase4-gpt-classification/scripts
```

### 步骤2: 检查GPU状态

```bash
nvidia-smi
```

### 步骤3: 运行实验

#### IMDB实验

```bash
# OELM-QK-FFN (3个学习率，分别在3个GPU上运行)
./run_imdb_oelm_qk_ffn.sh 0 0  # GPU 0, lr=1e-3
./run_imdb_oelm_qk_ffn.sh 1 1  # GPU 1, lr=3e-3
./run_imdb_oelm_qk_ffn.sh 2 2  # GPU 2, lr=1e-2

# OELM-FFN-only (3个学习率)
./run_imdb_oelm_ffn_only.sh 0 0  # GPU 0, lr=5e-4
./run_imdb_oelm_ffn_only.sh 1 1  # GPU 1, lr=1e-3
./run_imdb_oelm_ffn_only.sh 2 2  # GPU 2, lr=3e-3
```

#### AG News实验

```bash
./run_agnews_oelm_qk_ffn.sh 0 0
./run_agnews_oelm_qk_ffn.sh 1 1
./run_agnews_oelm_ffn_only.sh 0 0
./run_agnews_oelm_ffn_only.sh 1 1
```

#### XNLI实验

```bash
./run_xnli_oelm_qk_ffn.sh 0 0
./run_xnli_oelm_qk_ffn.sh 1 1
./run_xnli_oelm_ffn_only.sh 2 0
./run_xnli_oelm_ffn_only.sh 3 1
```

#### MNLI实验

```bash
./run_mnli_oelm_qk_ffn.sh 0 0
./run_mnli_oelm_qk_ffn.sh 1 1
./run_mnli_oelm_ffn_only.sh 2 0
./run_mnli_oelm_ffn_only.sh 3 1
```

### 后台运行（推荐用于长时间实验）

```bash
# 使用nohup在后台运行
nohup ./run_imdb_oelm_qk_ffn.sh 0 0 > /tmp/imdb_oelm_qk_ffn_0.log 2>&1 &

# 使用tmux（推荐）
tmux new -s imdb_oelm
cd /projects/Orthogonal_ELM_Transformers/Train/experiments/phase4-gpt-classification/scripts
./run_imdb_oelm_qk_ffn.sh 0 0

# 按Ctrl+B然后D分离会话
tmux attach -t imdb_oelm  # 重新连接
```

## 监控实验

### 查看运行中的进程

```bash
# 查看Python进程
ps aux | grep train_classification.py

# 查看GPU使用情况
watch -n 1 nvidia-smi
```

### 查看日志

```bash
# 实时查看日志
tail -f /projects/Orthogonal_ELM_Transformers/Train/outputs_phase4/IMDB_oelm_qk_ffn_lr1e-3/training.log

# 查看最后100行
tail -n 100 /projects/Orthogonal_ELM_Transformers/Train/outputs_phase4/IMDB_oelm_qk_ffn_lr1e-3/training.log
```

## 实验结果汇总

实验完成后，结果将保存在：

```
/projects/Orthogonal_ELM_Transformers/Train/outputs_phase4/
├── IMDB_oelm_qk_ffn_lr1e-3/
├── IMDB_oelm_qk_ffn_lr3e-3/
├── IMDB_oelm_ffn_only_lr5e-4/
├── AGNEWS_oelm_qk_ffn_lr1e-3/
└── ...
```

每个目录包含：
- `training.log`: 训练日志
- `config.json`: 实验配置
- `best_model.pt`: 最佳模型（如果保存了）
- `results.json`: 实验结果

## 下载结果到本地

```bash
# 下载整个outputs_phase4目录
rsync -avz tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/Train/outputs_phase4/ ./outputs_phase4/

# 下载特定实验结果
scp -r tianyu016@10.97.216.128:/projects/Orthogonal_ELM_Transformers/Train/outputs_phase4/IMDB_oelm_qk_ffn_lr1e-3 ./results/
```

## 实验计划建议

由于GPU资源有限，建议按以下顺序运行：

### 第一批（快速验证，预计6小时）
- IMDB OELM-QK-FFN (lr=1e-3, 3e-3)
- IMDB OELM-FFN-only (lr=5e-4, 1e-3)

### 第二批（确定最佳学习率后）
- AG News 最佳配置
- XNLI 最佳配置
- MNLI 最佳配置

### 并行策略
4个GPU可以同时运行4个实验：
- GPU 0: IMDB实验
- GPU 1: AG News实验
- GPU 2: XNLI实验
- GPU 3: MNLI实验

## 注意事项

1. **路径**: 所有脚本已配置为服务器路径 `/projects/Orthogonal_ELM_Transformers/Train`
2. **环境**: 使用服务器上的venv (`~/projects/oelm/venv`)
3. **存储**: 输出在 `/projects` 下（空间充足），不在 `/home` 下
4. **断线**: 使用 `tmux` 或 `nohup` 防止SSH断线导致实验中断
5. **冲突**: 避免在同一GPU上同时运行多个实验

## 故障排查

### 找不到模块

```bash
# 检查Python路径
cd /projects/Orthogonal_ELM_Transformers/Train
source ~/projects/oelm/venv/bin/activate
python -c "import sys; print(sys.path)"

# 测试模型导入
python -c "from experiments.phase4-gpt-classification.models.modeling_oelm_ffn import create_oelm_qk_ffn_classifier; print('OK')"
```

### GPU内存不足

```bash
# 减小batch size
python scripts/train_classification.py --batch_size 8 ...
```

### 权限问题

```bash
chmod +x scripts/*.sh
```
