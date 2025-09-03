#!/usr/bin/env bash
set -euo pipefail

echo "[setup_codex.sh] Starting setup..."

# --- System deps commonly needed by OpenCV/IMM ---
sudo apt-get update
DEBIAN_FRONTEND=noninteractive sudo apt-get install -y --no-install-recommends \
  build-essential git git-lfs cmake ninja-build \
  libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg
git lfs install

# --- Python toolchain ---
python -m pip install -U pip wheel setuptools

# --- Go to repo root ---
cd /workspace/colon_matching

# --- Init submodules just in case ---
git submodule update --init --recursive

# --- Install PyTorch (CPU wheel) ---
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# --- Install IMM (from source, recommended) ---
cd ../image-matching-models
pip install -e .
pip install '.[all]'
cd ../../

# --- Colon_matching extras (minimal set; extend if needed) ---
pip install -r requirements.txt

# --- Register kernel ---
python -m ipykernel install --user --name colon_matching --display-name "Python (colon_matching)"

# --- Quick smoke test ---
python - <<'PY'
from matching import get_matcher
m = get_matcher('sift-lg', device='cpu')
print("IMM OK:", type(m).__name__)
PY

echo "[setup_codex.sh] Done."
