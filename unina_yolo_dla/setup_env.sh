#!/bin/bash
# UNINA-YOLO-DLA: Virtual Environment Setup (Bash)
# =============================================================================

set -e

echo ">>> Creating virtual environment in .venv..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "    Virtual environment created."
else
    echo "    .venv already exists, skipping creation."
fi

echo ">>> Activating virtual environment..."
source .venv/bin/activate

echo ">>> Upgrading pip..."
pip install --upgrade pip

echo ">>> Installing core dependencies from requirements.txt..."
pip install -r requirements.txt

echo ">>> Installing NVIDIA Quantization tools..."
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com

echo ""
echo "============================================================"
echo ">>> SETUP COMPLETE!"
echo "============================================================"
echo "To start working, ensure the environment is active:"
echo "source .venv/bin/activate"
echo ""
