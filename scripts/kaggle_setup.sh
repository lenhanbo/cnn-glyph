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

echo "[2/4] Checking VATr dependencies (torch/torchvision are intentionally untouched)"
python - <<'PY'
import importlib
import subprocess
import sys

requirements = {
    "yaml": "PyYAML>=6.0",
    "cv2": "opencv-python-headless>=4.10",
    "pytest": "pytest>=7",
}
missing = []
for module, package in requirements.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "available")
        print(f"  {module}={version}")
    except ImportError:
        missing.append(package)
if missing:
    print("Installing missing packages:", ", ".join(missing))
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("  all dependencies already available; pip skipped")
PY

echo "[3/4] Locating raw IAM/IAM-32 input and validating CUDA"
python train_feature_map.py --config configs/kaggle.yaml --preflight-only "$@"

echo "[4/4] CPU contract tests"
python -m pytest -q

mkdir -p /kaggle/working/vatr-feature-map
echo "Setup complete. Start with:"
echo "python train_feature_map.py --config configs/kaggle.yaml --mode auto"
