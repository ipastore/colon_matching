# colon_matching

Tools for evaluating and benchmarking feature matching models on colonoscopy datasets.  
This repository depends on [image-matching-models (IMM)](https://github.com/alexstoken/image-matching-models) for local feature extraction and matching.

---

## 📦 Installation

We recommend a fresh Python 3.10+ environment (conda or venv).

### 1. Clone with submodules
```bash
git clone --recurse-submodules https://github.com/ipastore/colon_matching.git
cd colon_matching
````

If you already cloned without `--recurse-submodules`, run:

```bash
git submodule update --init --recursive
```

### 2. System dependencies (Linux / Ubuntu)

Make sure you have compilers and runtime libs for OpenCV:

```bash
sudo apt update
sudo apt install -y build-essential git git-lfs cmake ninja-build \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg
git lfs install
```

### 3. Python environment

Create and activate a new environment (example with conda):

```bash
conda create -n colon_matching python=3.10 -y
conda activate colon_matching
```

### 4. Install PyTorch

Pick one (see [PyTorch install](https://pytorch.org/get-started/locally/)):

CPU only:

```bash
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
```

CUDA 12.1 (if GPU available):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio
```

### 5. Install IMM

From the submodule:

```bash
cd utils/image-matching-models
pip install -e .
pip install '.[all]'   # optional: install extras (LoFTR, OmniGlue, etc.)
cd ../../
```

### 6. Colon Matching extras

```bash
pip install ipykernel pandas matplotlib scikit-learn scikit-image tqdm open3d-cpu pycolmap plotly dash
```

### 7. (Optional) Jupyter kernel

```bash
python -m ipykernel install --user --name colon_matching --display-name "Python (colon_matching)"
```

---

## ✅ Quick test

Verify IMM is working:

```bash
python - <<'PY'
from matching import get_matcher
m = get_matcher('sift-lg', device='cpu')
print("IMM OK:", type(m).__name__)
PY
```

Expected output:

```
IMM OK: <some matcher class>
```

---

## 🚀 Usage

### Benchmarking

Run the main benchmarking script:

```bash
python benchmark.py --config configs/example.yaml
```

### Demo notebook

Launch Jupyter and open:

```bash
jupyter notebook demo.ipynb
```

---

## 🗂 Project structure

```
colon_matching/
│
├── tools/
│   └── image-matching-models/   # IMM submodule
├── configs/                     # Example configs for experiments
├── benchmark.py                 # Main benchmarking script
├── demo.ipynb                   # Example notebook
└── scripts/                     # Helper setup and automation scripts
```

---

## ⚠️ Notes

* IMM requires **PyTorch ≥ 2.2**.
* Some models (LoFTR family, OmniGlue) need extra deps like `tensorflow` and `pytorch-lightning`; install with `[all]`.
* `open3d`, `vtk`, and `pycolmap` are optional but useful for visualization and 3D reconstruction.

---

## 🛠 Development

To keep your environment up to date:

```bash
git pull
git submodule update --init --recursive
pip install -e utils/image-matching-models
```
