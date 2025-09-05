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
cd "$(git rev-parse --show-toplevel)"
pwd
ls -lah

# Install dependencies of main repo
pip install -r "requirements.txt"

# --- Init submodules just in case ---
cd "$(git rev-parse --show-toplevel)"
ls -lah

git lfs install --skip-smudge

# git config submodule.matching/third_party/SphereGlue.update none
git submodule update --init --recursive tools/image-matching_models

# --- Install PyTorch (CPU wheel) ---
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

# --- DEBUG ---
echo "Contents of tools/:"
ls -l tools/
echo "current wd"
pwd

# --- Install IMM (from source, recommended) ---
cd tools/image-matching_models
echo "Installing IMM..."
pip install -e .
echo "Installing IMM with all extras..."
pip install '.[all]'
echo "IMM installed with all extras."

echo "Registering ipkernel..."
# --- Register kernel ---
python -m ipykernel install --user --name colon_matching --display-name "Python (colon_matching)"

echo "[setup_codex.sh] Done."
