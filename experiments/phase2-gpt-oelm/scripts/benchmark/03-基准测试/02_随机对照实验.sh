#!/bin/bash
# Run random Gaussian control experiment
# This script trains a model with frozen random Gaussian Q/K projections
# (instead of orthogonal) as a control experiment.

# Note: This requires modifying the model to use Gaussian initialization
# This script demonstrates the experimental setup

echo "=========================================="
echo "Random Gaussian Control Experiment"
echo "=========================================="
echo "This experiment trains a model with frozen"
echo "random Gaussian Q/K projections to compare"
echo "with orthogonal initialization."
echo "=========================================="

# For now, this is a placeholder. To implement:
# 1. Create a variant of OrthogonalLinear that uses Gaussian initialization
# 2. Train with the same hyperparameters as OELM
# 3. Compare results

echo "To implement this experiment:"
echo "1. Create models/modeling_random_control.py"
echo "2. Implement GaussianLinear layer"
echo "3. Run training with --model_type random_control"
