#!/usr/bin/env bash
# Creates an isolated Python 3.12 venv with TensorFlow for model conversion.
# Run once from the repo root:  bash scripts/setup_tf_env.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/.venv-tf"

echo "==> Creating Python 3.12 venv at $VENV_DIR"
uv venv "$VENV_DIR" --python 3.12

echo "==> Installing TensorFlow 2.17 (CPU build, ~600 MB)"
uv pip install \
  --python "$VENV_DIR/bin/python" \
  -r "$REPO_ROOT/scripts/requirements-convert.txt"

echo ""
echo "Done. Activate with:"
echo "  source $VENV_DIR/bin/activate"
echo ""
echo "Then run:"
echo "  python scripts/convert_to_tflite.py"
echo "  python scripts/test_tflite_model.py"
