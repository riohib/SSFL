#!/bin/bash
# setup.sh - Robust setup for SSFL project

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "🚀 Setting up SSFL environment..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ Error: uv is not installed"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Remove old environment if it exists
if [ -d ".venv" ]; then
    echo "⚠️  Removing old '.venv' environment..."
    rm -rf .venv
fi

# Create new virtual environment with Python 3.10
echo "📦 Creating virtual environment (Python 3.10)..."
uv venv .venv --python 3.10

# Activate the environment for subsequent commands
source .venv/bin/activate

# Install PyTorch FIRST with CUDA support
# Default to CUDA 12.1, but can be changed via CUDA_VERSION env var
CUDA_VERSION=${CUDA_VERSION:-cu128}
echo "🔥 Installing PyTorch with CUDA support (${CUDA_VERSION})..."
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/${CUDA_VERSION}

# Install project dependencies from pyproject.toml
echo "📦 Installing project dependencies..."
uv pip install -e .

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
python -c "import numpy; print(f'NumPy {numpy.__version__}')"
python -c "import wandb; print(f'Wandb {wandb.__version__}')"
python -c "import omegaconf; print(f'OmegaConf {omegaconf.__version__}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "💡 To use the environment:"
echo "   source .venv/bin/activate"
echo ""
echo "💡 To change CUDA version, set CUDA_VERSION env var:"
echo "   CUDA_VERSION=cu128 bash setup.sh"
echo ""
