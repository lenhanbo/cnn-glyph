#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

ACTION="${1:-auto}"
case "$ACTION" in
  setup)
    exec bash scripts/kaggle_setup.sh "${@:2}"
    ;;
  smoke)
    exec python train_feature_map.py --config configs/kaggle.yaml --mode auto --smoke "${@:2}"
    ;;
  resume)
    exec python train_feature_map.py --config configs/kaggle.yaml --resume "${@:2}"
    ;;
  auto|stage1|reference_baseline)
    exec python train_feature_map.py --config configs/kaggle.yaml --mode "$ACTION" "${@:2}"
    ;;
  stage2)
    if [[ $# -lt 2 ]]; then
      echo "Usage: bash scripts/kaggle_run.sh stage2 /path/to/stage1-checkpoint.pth" >&2
      exit 2
    fi
    exec python train_feature_map.py --config configs/kaggle.yaml --mode stage2 --stage1-checkpoint "$2" "${@:3}"
    ;;
  *)
    echo "Unknown action: $ACTION (setup|smoke|auto|stage1|stage2|reference_baseline|resume)" >&2
    exit 2
    ;;
esac
