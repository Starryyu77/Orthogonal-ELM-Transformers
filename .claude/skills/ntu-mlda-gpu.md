# NTU MLDA GPU 服务器使用技能

## 概述

这个 skill 提供对 NTU MLDA GPU 服务器集群的快速访问和操作能力，支持在本地终端通过快捷命令控制远程服务器上的深度学习训练任务。

## 服务器信息

| 配置 | 详情 |
|-----|------|
| **服务器地址** | `gpu43.dynip.ntu.edu.sg` |
| **用户名** | `s125mdg43_10` |
| **密码** | `ADeaTHStARB` |
| **SSH 别名** | `ntu-gpu43` |
| **GPU 配置** | 4x NVIDIA RTX A5000 (24GB 显存) |
| **CUDA 版本** | 12.2 |
| **可用 GPU** | GPU 2, 3 (GPU 0,1 通常被其他用户占用) |

## 前置要求

1. **SSH 密钥配置完成** (首次使用需要设置)
2. **本地快捷脚本** `ntu-run.sh` 在项目目录中
3. **服务器环境** 已安装 Python + PyTorch + CUDA

## 快捷命令参考

### 基础操作

```bash
# 查看 GPU 状态
./ntu-run.sh status

# 检查 Python/PyTorch 环境
./ntu-run.sh check

# 同步代码到服务器
./ntu-run.sh sync

# 进入服务器交互式终端
./ntu-run.sh bash
```

### 训练操作

```bash
# 单卡训练 (使用 GPU 2)
./ntu-run.sh train <model_type> <config>
# 示例: ./ntu-run.sh train oelm small

# 多卡训练 (使用 GPU 2,3)
./ntu-run.sh train-multi <model_type> <num_gpus>
# 示例: ./ntu-run.sh train-multi oelm 2

# 运行基准测试
./ntu-run.sh benchmark <size>
# 示例: ./ntu-run.sh benchmark small
```

### 开发工具

```bash
# 启动 Jupyter Lab (本地访问 http://localhost:8888)
./ntu-run.sh jupyter

# 启动 TensorBoard (本地访问 http://localhost:6006)
./ntu-run.sh tensorboard

# 运行任意 Python 命令
./ntu-run.sh python <script.py> [args]
# 示例: ./ntu-run.sh python train.py --help
```

## 在新项目中使用

### 第一步：复制脚本

```bash
# 从已有项目复制 ntu-run.sh 到新项目
cp /path/to/oelm-project/ntu-run.sh /path/to/new-project/
cd /path/to/new-project
chmod +x ntu-run.sh
```

### 第二步：修改配置（可选）

编辑 `ntu-run.sh` 中的项目特定配置：

```bash
# 默认训练参数
DEFAULT_BATCH_SIZE=4
DEFAULT_STEPS=10000
DEFAULT_DATA="data/tinystories/train.bin"
```

### 第三步：准备服务器环境

```bash
# SSH 登录服务器
ssh ntu-gpu43

# 创建项目目录
mkdir -p ~/projects/<new-project>
cd ~/projects/<new-project>

# 安装虚拟环境
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install <其他依赖>

# 退出服务器
exit
```

### 第四步：同步代码并开始训练

```bash
# 本地终端：同步代码
./ntu-run.sh sync

# 开始训练
./ntu-run.sh train-multi <model> 2
```

## 直接 SSH 操作

当快捷命令不满足需求时，直接 SSH：

```bash
# 连接
ssh ntu-gpu43

# 激活环境
cd ~/projects/<project-name>
source venv/bin/activate
export CUDA_VISIBLE_DEVICES=2,3

# 手动运行训练
torchrun --standalone --nproc_per_node=2 train.py \
    --model_type oelm \
    --data_path data/tinystories/train.bin \
    --batch_size 4 \
    --max_steps 10000
```

## 文件传输

```bash
# 上传文件到服务器
scp local_file ntu-gpu43:~/projects/<project>/

# 下载文件到本地
scp ntu-gpu43:~/projects/<project>/best.pt ./

# 同步整个目录
rsync -avz --progress ./ ntu-gpu43:~/projects/<project>/
```

## 监控训练

```bash
# 查看 GPU 实时状态
watch -n 1 ./ntu-run.sh status

# 查看进程
ssh ntu-gpu43 "ps aux | grep train.py | grep -v grep"

# 查看输出文件
ssh ntu-gpu43 "ls -lh ~/projects/<project>/out/"
```

## 故障排除

### 连接失败
```bash
# 检查 SSH 配置
cat ~/.ssh/config | grep -A 5 "ntu-gpu43"

# 手动测试连接
ssh s125mdg43_10@gpu43.dynip.ntu.edu.sg
```

### 显存不足 (OOM)
- 减小 batch_size (在 ntu-run.sh 中修改)
- 使用单卡训练: `./ntu-run.sh train oelm small`

### 权限问题
```bash
# 确保脚本可执行
chmod +x ntu-run.sh

# 检查服务器目录权限
ssh ntu-gpu43 "ls -la ~/projects/"
```

## 最佳实践

1. **总是在本地编辑代码**，然后 `./ntu-run.sh sync` 同步
2. **定期保存模型检查点**，防止训练中断
3. **使用 `screen` 或 `tmux`** 在服务器上保持长时间任务
4. **训练完成后下载模型** `scp ntu-gpu43:~/projects/<project>/out/*/best.pt ./`

## 联系支持

- MLDA 集群问题: 联系 NTU MLDA 管理员
- 账户问题: `s125mdg43_10` 是共享账户，注意使用规范
