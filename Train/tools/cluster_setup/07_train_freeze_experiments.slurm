#!/bin/bash
# 冻结比例实验脚本
# 提交不同冻结比例的OELM模型训练

# 25% 冻结
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=oelm_fr25
#SBATCH --time=24:00:00
#SBATCH --gpus=v100:2
#SBATCH --output=logs/oelm_fr25_%j.out
#SBATCH --error=logs/oelm_fr25_%j.err
module load Miniconda3 && source activate oelm
cd /projects/oelm-experiment
python -m torch.distributed.run --nproc_per_node=2 scripts/02-训练脚本/train.py \
    --model_type oelm --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 32 --max_steps 100000 --freeze_ratio 0.25 \
    --data_path data/wikitext103/train.bin --val_data_path data/wikitext103/val.bin \
    --out_dir models/checkpoints/oelm_fr25 --use_wandb --wandb_run_name oelm_fr25
EOF

# 50% 冻结
sbatch << 'EOF'
#!/bin/bash
#SBATCH --job-name=oelm_fr50
#SBATCH --time=24:00:00
#SBATCH --gpus=v100:2
#SBATCH --output=logs/oelm_fr50_%j.out
#SBATCH --error=logs/oelm_fr50_%j.err
module load Miniconda3 && source activate oelm
cd /projects/oelm-experiment
python -m torch.distributed.run --nproc_per_node=2 scripts/02-训练脚本/train.py \
    --model_type oelm --d_model 512 --num_layers 6 --num_heads 8 --d_ff 2048 \
    --seq_len 512 --batch_size 32 --max_steps 100000 --freeze_ratio 0.50 \
    --data_path data/wikitext103/train.bin --val_data_path data/wikitext103/val.bin \
    --out_dir models/checkpoints/oelm_fr50 --use_wandb --wandb_run_name oelm_fr50
EOF

echo "冻结比例实验已提交: 25%, 50%"
