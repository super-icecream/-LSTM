#!/bin/bash
# DLFE-LSTM-WSI environment setup script (GPU/CPU)
# Creates the conda env at a fixed prefix: <repo>/.conda

set -euo pipefail

YAML_FILE="environment/environment.yml"
REQ_FILE="environment/requirements.txt"

usage() {
  echo "Usage: $0 [--cpu]"
  echo "  --cpu   install CPU-only PyTorch"
}

USE_CPU=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu) USE_CPU=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found. Please install Anaconda or Miniconda."
  exit 1
fi

# Resolve project root and env prefix
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PREFIX="${PROJECT_ROOT}/.conda"

echo "Creating conda environment at prefix: ${ENV_PREFIX}"
conda env remove -p "${ENV_PREFIX}" -y >/dev/null 2>&1 || true
conda env create -f "${YAML_FILE}" -p "${ENV_PREFIX}"

# Activate the new env
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

if ${USE_CPU}; then
  echo "Switching to CPU-only PyTorch wheels"
  pip uninstall -y torch torchvision torchaudio || true
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

echo "Installing extra pip dependencies"
pip install -r "${REQ_FILE}"
pip install pytest pytest-cov black flake8 mypy

echo "Verifying core dependencies"
python - <<'PY'
import torch, numpy, pandas, plotly
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"Plotly: {plotly.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
PY

echo "Environment setup complete."
echo "Activate with: conda activate \"C:\\Users\\Administrator\\桌面\\专利\\DLFE-LSTM-WSI\\.conda\""

