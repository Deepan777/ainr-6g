#!/usr/bin/env bash
set -e

echo "=== AINR-6G Environment Setup ==="

python -m venv venv
source venv/bin/activate

pip install --upgrade pip

# PyTorch 2.9.1 with CUDA 12.6 (satisfies sionna 2.x requirement)
pip install torch==2.9.1+cu126 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu126

pip install -r requirements.txt

echo "=== Verifying imports ==="
python -c "import sionna; import torch; import pyro; print('OK — sionna', sionna.__version__, 'torch', torch.__version__, 'pyro', pyro.__version__)"

echo "=== Setup complete. Activate with: source venv/bin/activate ==="
