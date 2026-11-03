#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "[1/4] Python and GPU"
python --version
python - <<'PY'
import torch, torchvision
print(f"torch={torch.__version__} torchvision={torchvision.__version__}")
print(f"cuda={torch.cuda.is_available()} device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
PY

echo "[2/4] Installing VATr dependencies (torch/torchvision are intentionally absent)"
python -m pip install -q -r requirements-kaggle.txt

echo "[3/4] Locating raw IAM/IAM-32 input and validating CUDA"
python train_feature_map.py --config configs/kaggle.yaml --preflight-only "$@"

echo "[4/4] CPU contract tests"
python -m pytest -q

mkdir -p /kaggle/working/vatr-feature-map
echo "Setup complete. Start with:"
echo "python train_feature_map.py --config configs/kaggle.yaml --mode auto"
