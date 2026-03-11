#!/bin/bash
#SBATCH --job-name=oelm-finetune
#SBATCH --time=4:00:00
#SBATCH --gpus=pro6000:1
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err

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

# Arguments
PRETRAINED_PATH=$1
DATASET=${2:-imdb}
METHOD=${3:-oelm_qk}

# Print job info
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Pretrained: $PRETRAINED_PATH"
echo "Dataset: $DATASET"
echo "Method: $METHOD"
echo "Start time: $(date)"
echo "========================================"

# Run fine-tuning
python scripts/finetune_from_pretrain.py \
    --pretrained_path $PRETRAINED_PATH \
    --dataset $DATASET \
    --freeze_qk \
    --batch_size 16 \
    --num_epochs 3 \
    --lr 1e-3 \
    --output_dir ./outputs/finetune/${METHOD}_${DATASET} \
    --gpu 0

echo "========================================"
echo "End time: $(date)"
echo "========================================"