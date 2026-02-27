# 快速开始指南

## 1. 环境准备

```bash
# 克隆或复制本文件夹到目标机器
cd bert_experiments

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt
```

## 2. 快速测试

运行快速测试确保环境配置正确（约2-3分钟）：

```bash
bash test_setup.sh
```

如果所有测试通过，说明环境配置正确。

## 3. 运行实验

### 方式A: 批量运行所有实验（推荐）

```bash
bash run_all_experiments.sh
```

这会在后台启动所有4个实验，日志保存在`logs/`目录。

### 方式B: 单独运行某个实验

```bash
# IMDB
python bert_imdb_experiments.py

# AG News
python bert_agnews_experiments.py

# MNLI
python bert_mnli_experiments.py

# XNLI (英语)
python bert_xnli_experiments.py --language en
```

## 4. 监控进度

```bash
# 查看实时日志
tail -f logs/imdb_log.txt

# 查看所有运行的实验
ps aux | grep bert_.*_experiments.py | grep -v grep
```

## 5. 查看结果

实验完成后，结果会保存在JSON文件中：
- `bert_imdb_results.json`
- `bert_agnews_results.json`
- `bert_mnli_results.json`
- `bert_xnli_en_results.json`

## 预计运行时间

| 数据集 | 训练样本 | 单次实验 | 10次重复 |
|--------|---------|---------|----------|
| IMDB   | 25,000  | ~14分钟 | ~2.3小时 |
| AG News| 120,000 | ~25分钟 | ~4.1小时 |
| MNLI   | 392,702 | ~60分钟 | ~10小时  |
| XNLI   | 392,702 | ~60分钟 | ~10小时  |

**总计**: 约26小时（顺序运行）或 ~10小时（并行运行，需要足够GPU内存）

## 故障排查

### GPU内存不足

减小批次大小：
```bash
python bert_xxx_experiments.py --batch-size 8
```

### 数据集下载慢

使用Hugging Face镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Python包缺失

```bash
pip install -r requirements.txt --upgrade
```

## 更多信息

详细说明请查看 [README.md](README.md)
