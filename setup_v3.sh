#!/bin/bash

# Setup script for CogVideoX V3 Experiment
# This script installs all required dependencies

echo "=========================================="
echo "CogVideoX V3 Experiment Setup"
echo "=========================================="

# Install torchmetrics (required for CLIP Score)
echo "Installing torchmetrics..."
pip install torchmetrics>=1.0.0

# Install other dependencies if needed
echo "Installing other dependencies..."
pip install -r requirements_v3.txt

echo ""
echo "=========================================="
echo "✓ Setup complete!"
echo "=========================================="
echo ""
echo "You can now run the experiment:"
echo "  python experiment_cogvideox_v3.py"
echo ""
