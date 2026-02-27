#!/bin/bash
# MLDA GPU 训练脚本 - OELM Medium-512
# 配置: n_layer=6, d_model=512, n_head=8, freeze_ratio=0.075

echo "=== OELM Medium-512 训练 (MLDA GPU) ==="
echo "开始时间: $(date)"

# 进入项目目录
cd ~/Orthogonal_ELM_Transformers/Train

# 激活conda环境
source ~/miniconda3/bin/activate oelm

# 设置环境变量
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# 训练命令
python -m torch.distributed.run \
    --nproc_per_node=2 \
    --master_port=29501 \
    scripts/02-训练脚本/train.py \
    --model_type oelm \
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
    --freeze_ratio 0.075 \
    --data_path data/tiny_stories/train.bin \
    --val_data_path data/tiny_stories/val.bin \
    --val_steps 1000 \
    --checkpoint_steps 5000 \
    --out_dir models/checkpoints/oelm_medium512 \
    --use_wandb \
    --wandb_project oelm-experiment \
    --wandb_run_name oelm_medium512_mlda \
    --compile

echo "训练完成: $(date)"
