#!/bin/bash
#SBATCH --job-name=gpt_medium512
#SBATCH --time=24:00:00
#SBATCH --gpus=v100:2
#SBATCH --output=logs/gpt_%j.out
#SBATCH --error=logs/gpt_%j.err
#SBATCH --partition=cluster02
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=tianyu016@e.ntu.edu.sg

# 同尺寸GPT基线训练
# 配置: n_layer=6, d_model=512, n_head=8

echo "=== GPT Medium-512 基线训练 ==="
echo "开始时间: $(date)"
echo "作业ID: $SLURM_JOB_ID"
echo "节点: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# 激活venv环境
source /projects/oelmexperiment/.venv/bin/activate

# 进入项目目录
cd /projects/oelmexperiment

# 设置环境变量
export PYTHONUNBUFFERED=1

# 训练命令
python -m torch.distributed.run \
    --nproc_per_node=2 \
    scripts/02-训练脚本/train.py \
    --model_type gpt \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --seq_len 512 \
    --batch_size 32 \
    --gradient_accumulation_steps 2 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --data_path data/wikitext103/train.bin \
    --val_data_path data/wikitext103/val.bin \
    --val_steps 1000 \
    --checkpoint_steps 5000 \
    --out_dir models/checkpoints/gpt_medium512_wikitext \
    --use_wandb \
    --wandb_project oelm-experiment \
    --wandb_run_name gpt_medium512_baseline

echo "训练完成: $(date)"
