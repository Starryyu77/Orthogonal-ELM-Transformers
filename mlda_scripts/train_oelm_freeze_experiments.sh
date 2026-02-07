#!/bin/bash
# MLDA GPU 冻结比例实验脚本
# 测试不同 freeze_ratio: 0.25, 0.50

echo "=== OELM 冻结比例实验 (MLDA GPU) ==="

cd ~/Orthogonal_ELM_Transformers/Train
source ~/miniconda3/bin/activate oelm

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0,1

# 实验1: freeze_ratio=0.25 (25%冻结)
echo "启动实验1: freeze_ratio=0.25"
python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29502 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 32 --max_steps 100000 --freeze_ratio 0.25 \
    --data_path data/tiny_stories/train.bin --val_data_path data/tiny_stories/val.bin \
    --out_dir models/checkpoints/oelm_fr25 --wandb_run_name oelm_fr25_mlda &

# 实验2: freeze_ratio=0.50 (50%冻结)
echo "启动实验2: freeze_ratio=0.50"
python -m torch.distributed.run \
    --nproc_per_node=2 --master_port=29503 \
    scripts/02-训练脚本/train.py \
    --model_type oelm --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 32 --max_steps 100000 --freeze_ratio 0.50 \
    --data_path data/tiny_stories/train.bin --val_data_path data/tiny_stories/val.bin \
    --out_dir models/checkpoints/oelm_fr50 --wandb_run_name oelm_fr50_mlda &

wait
echo "所有实验完成: $(date)"
