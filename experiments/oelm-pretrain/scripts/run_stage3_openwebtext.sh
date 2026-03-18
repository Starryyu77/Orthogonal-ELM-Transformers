#!/bin/bash
#SBATCH --job-name=oelm-stage3-openwebtext
#SBATCH --partition=cluster02
#SBATCH --gpus=pro-6000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/stage3_%j.out
#SBATCH --error=logs/stage3_%j.err

# OELM Stage 3: OpenWebText Medium-scale Pretraining
# Usage: sbatch scripts/run_stage3_openwebtext.sh [baseline|oelm_qk|oelm_qk_ffn]

METHOD=${1:-baseline}

echo "=========================================="
echo "Stage 3: OpenWebText Pretraining"
echo "Method: $METHOD"
echo "Started: $(date)"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd /projects/LlamaFactory/OELM-Pretrain

source ~/miniconda3/etc/profile.d/conda.sh
conda activate oelm

# Method configuration
case $METHOD in
    baseline)
        FREEZE_QK="false"
        FREEZE_FFN="false"
        LR="3e-4"
        ;;
    oelm_qk)
        FREEZE_QK="true"
        FREEZE_FFN="false"
        LR="3e-4"
        ;;
    oelm_qk_ffn)
        FREEZE_QK="true"
        FREEZE_FFN="true"
        LR="3e-4"
        ;;
    *)
        echo "Unknown method: $METHOD"
        exit 1
        ;;
esac

# DDP training with 2 GPUs
torchrun \
    --nproc_per_node=2 \
    --master_port=29500 \
    scripts/pretrain.py \
    --dataset openwebtext \
    --model_name gpt2 \
    --max_steps 100000 \
    --batch_size 128 \
    --gradient_accumulation_steps 2 \
    --seq_length 1024 \
    --learning_rate $LR \
    --lr_scheduler_type cosine \
    --warmup_steps 5000 \
    --freeze_qk $FREEZE_QK \
    --freeze_ffn $FREEZE_FFN \
    --output_dir outputs/stage3_openwebtext/$METHOD \
    --save_steps 10000 \
    --eval_steps 5000 \
    --logging_steps 100 \
    --fp16 \
    --ddp_find_unused_parameters false

echo "=========================================="
echo "Completed: $(date)"
echo "Output: outputs/stage3_openwebtext/$METHOD"
echo "=========================================="
