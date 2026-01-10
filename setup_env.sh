#!/bin/bash
# ============================================================================
# Environment Setup Script
# ============================================================================
# Phase 0: Local M1 MacBook (API-based LLM/VLM)
# Phase 1+: Nvidia GPU (Local LLM/VLM inference/finetuning)
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ============================================================================

set -e  # Exit on error

echo "============================================"
echo "Environment Setup"
echo "============================================"

# Detect OS
OS="$(uname -s)"
echo "Detected OS: $OS"

# ============================================================================
# PHASE 0: Basic Setup (M1 Mac / Any Platform)
# ============================================================================

echo ""
echo "[Phase 0] Basic Python Environment Setup"
echo "============================================"

# Create virtual environment
if [ ! -d "venv_3Denv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_3Denv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv_3Denv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install Phase 0 requirements
echo "Installing Phase 0 requirements..."
pip install -r requirements.txt

echo ""
echo "============================================"
echo "Phase 0 Setup Complete!"
echo "============================================"
echo ""
echo "To activate environment:"
echo "  source venv_3Denv/bin/activate"
echo ""
echo "To run Phase 0 tests:"
echo "  python src/m01_shortest_path.py --test"
echo "  python src/m02_hybrid_judge.py --test"
echo ""

# ============================================================================
# PHASE 1+: GPU Setup (Nvidia GPU - Linux)
# Uncomment when ready for local LLM/VLM inference/finetuning
# ============================================================================

# # Check if running on Linux with GPU
# if [ "$OS" = "Linux" ]; then
#     echo ""
#     echo "[Phase 1+] GPU Environment Setup (Linux + Nvidia)"
#     echo "============================================"
#
#     # 0. Install LINUX system dependencies
#     echo ""
#     echo "[0/5] Installing system dependencies..."
#     sudo apt update && sudo apt install -y \
#         texlive-latex-base \
#         texlive-latex-extra \
#         texlive-fonts-recommended \
#         texlive-fonts-extra \
#         texlive-bibtex-extra \
#         texlive-science \
#         biber \
#         tree
#
#     # 1. Install Python 3.12 (if not already installed)
#     echo ""
#     echo "[1/6] Installing Python 3.12..."
#     sudo apt install -y software-properties-common
#     sudo add-apt-repository -y ppa:deadsnakes/ppa
#     sudo apt update
#     sudo apt install -y python3.12 python3.12-venv python3.12-dev
#
#     # 2. Install PyTorch 2.5.1 with CUDA 12.4
#     echo ""
#     echo "[2/6] Installing PyTorch 2.5.1+cu124..."
#     pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
#
#     # 3. Verify PyTorch installation
#     echo ""
#     echo "[3/6] Verifying PyTorch..."
#     python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, Available: {torch.cuda.is_available()}')"
#
#     # 4. Install GPU requirements (add to requirements_gpu.txt when needed)
#     # echo ""
#     # echo "[4/6] Installing GPU requirements..."
#     # pip install -r requirements_gpu.txt
#
#     # 5. Install Flash-Attention (pre-built wheel for torch2.5+cu12+cp312)
#     echo ""
#     echo "[5/6] Installing Flash-Attention 2.8.3..."
#     WHEEL_NAME="flash_attn-2.8.3+cu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
#     WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.5cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
#
#     curl -L -o "$WHEEL_NAME" "$WHEEL_URL"
#     pip install "$WHEEL_NAME"
#     rm -f "$WHEEL_NAME"
#
#     # 6. Final verification
#     echo ""
#     echo "[6/6] Verifying GPU setup..."
#     python -c "
# import torch
# import flash_attn
# print(f'PyTorch:    {torch.__version__}')
# print(f'CUDA:       {torch.version.cuda}')
# print(f'GPU:        {torch.cuda.is_available()}')
# print(f'Flash-Attn: {flash_attn.__version__}')
# "
#
#     echo ""
#     echo "============================================"
#     echo "GPU Setup Complete!"
#     echo "============================================"
# fi
