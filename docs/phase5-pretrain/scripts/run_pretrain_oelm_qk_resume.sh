#!/bin/bash
#SBATCH --job-name=oelm-qk-resume
#SBATCH --time=12:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --output=logs/pretrain_oelm_qk_resume-%j.out
#SBATCH --error=logs/pretrain_oelm_qk_resume-%j.err
#SBATCH --constraint=gpu_48g

module load Miniforge3
source activate
cd /projects/LlamaFactory/OELM-Pretrain

echo "========================================"
echo "Resuming OELM-QK from checkpoint-104000"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "========================================"

python scripts/train_pretrain.py \
    --model_size small \
    --method oelm_qk \
    --dataset tinystories \
    --batch_size 16 \
    --num_epochs 1 \
    --max_seq_len 512 \
    --lr 1e-3 \
    --save_steps 2000 \
    --output_dir ./outputs/pretrain \
    --gpu 0 \
    --num_workers 4 \
    --resume_from outputs/pretrain/oelm_qk/checkpoint-104000/pytorch_model.pt

echo "========================================"
echo "End time: $(date)"
echo "========================================"