#!/bin/bash
# Setup conda environment for OELM experiments
# Run this once on the cluster

set -e

echo "========================================"
echo "Setting up OELM environment"
echo "========================================"

# Load miniforge
module load Miniforge3
source activate

# Create environment
ENV_PATH="/projects/LlamaFactory/.conda/envs/oelm"

if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH"
    read -p "Recreate? (y/n): " confirm
    if [ "$confirm" = "y" ]; then
        conda env remove -p $ENV_PATH -y
    else
        echo "Using existing environment"
        conda activate $ENV_PATH
        exit 0
    fi
fi

# Create new environment
echo "Creating new environment..."
conda create -p $ENV_PATH python=3.10 -y

# Activate
conda activate $ENV_PATH

# Install dependencies
echo "Installing dependencies..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets tqdm scikit-learn numpy

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "========================================"
echo "Environment setup complete!"
echo "========================================"
echo "Activate with: conda activate $ENV_PATH"