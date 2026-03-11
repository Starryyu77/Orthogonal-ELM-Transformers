#!/bin/bash
#SBATCH --job-name=oelm-pretrain-qk
#SBATCH --time=24:00:00
#SBATCH --gpus=pro6000:2
#SBATCH --output=logs/pretrain_oelm_qk-%j.out
#SBATCH --error=logs/pretrain_oelm_qk-%j.err
#SBATCH --constraint=gpu_48g

# Load environment
module load Miniforge3
source activate

# Activate conda environment
if [ -d "/projects/LlamaFactory/.conda/envs/oelm" ]; then
    conda activate /projects/LlamaFactory/.conda/envs/oelm
else
    conda activate base
    pip install torch transformers datasets tqdm scikit-learn
fi

# Set working directory
cd /projects/LlamaFactory/OELM-Pretrain

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

# Run training
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
    --num_workers 4

echo "========================================"
echo "End time: $(date)"
echo "========================================"