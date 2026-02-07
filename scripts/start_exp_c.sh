#!/bin/bash
# 启动 Group C (OELM-Freeze) 实验
# 在 Group A 和 B 完成后运行此脚本

cd ~/Orthogonal_ELM_Transformers/Train

# Kill existing screen if exists
screen -S exp_oelm_f -X quit 2>/dev/null
sleep 2

# Start Group C
screen -dmS exp_oelm_f bash -c "
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTHONPATH=/usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train:\$PYTHONPATH
source ~/projects/oelm/venv/bin/activate
cd /usr1/home/s125mdg43_10/Orthogonal_ELM_Transformers/Train

python -m torch.distributed.run \
    --nproc_per_node=4 \
    --master_port=29502 \
    scripts/02-训练脚本/train.py \
    --model_type oelm \
    --freeze_qk true \
    --d_model 512 \
    --num_layers 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --seq_len 512 \
    --batch_size 8 \
    --max_steps 100000 \
    --warmup_steps 2000 \
    --max_lr 3e-4 \
    --min_lr 3e-5 \
    --weight_decay 0.1 \
    --grad_clip 1.0 \
    --beta1 0.9 \
    --beta2 0.95 \
    --data_path data/tiny_stories/train.bin \
    --log_interval 100 \
    --val_interval 1000 \
    --save_interval 5000 \
    --out_dir models/checkpoints/exp_oelm_freeze \
    --wandb_run_name exp_oelm_freeze \
    2>&1 | tee models/checkpoints/exp_oelm_freeze/training.log

exec bash
"

echo "Group C (OELM-Freeze) started"
echo "Screen session: exp_oelm_f"
echo "Monitor with: screen -r exp_oelm_f"
