# Environment Setup

Both platforms use **Miniforge** (conda) + **Spyder**. The steps are nearly identical.

---

## 1. Install Miniforge

Miniforge is a minimal conda installer that defaults to the `conda-forge` channel — lighter than Anaconda and works the same on both OSes.

**Linux (Mint / Ubuntu)**
```bash
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
# Restart your terminal after install
```

**Windows**
- Download the installer from: https://github.com/conda-forge/miniforge/releases/latest
- File: `Miniforge3-Windows-x86_64.exe`
- Run it and accept defaults. Check "Add to PATH" if prompted (makes things easier).
- After install, open **Miniforge Prompt** from the Start menu (use this instead of regular Command Prompt).

---

## 2. Create the environment

Run this from the project root (where `environment.yml` lives).

**Linux**
```bash
cd /path/to/MolGraph
conda env create -f environment.yml
```

**Windows** (in Miniforge Prompt)
```
cd C:\path\to\MolGraph
conda env create -f environment.yml
```

This takes a few minutes — PyTorch and PyG are large.

---

## 3. Activate the environment

```bash
conda activate molgraph
```

You need to do this every time you open a new terminal before launching Spyder.

---

## 4. Launch Spyder

```bash
spyder
```

Spyder will open. The Python interpreter in the bottom-right corner should show `molgraph`.

**If Spyder opens but can't import torch/rdkit**, it's using the wrong kernel. Fix:
- In Spyder: `Tools → Preferences → Python interpreter`
- Select "Use the following Python interpreter"
- Point it to the conda env's Python:
  - Linux: `~/miniforge3/envs/molgraph/bin/python`
  - Windows: `C:\Users\<you>\miniforge3\envs\molgraph\python.exe`
- Restart the kernel (`Consoles → Restart kernel`)

---

## 5. Verify the install

Paste this into the Spyder console and run it:

```python
import torch
import torch_geometric
import rdkit
import numpy, pandas, sklearn, matplotlib

print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("RDKit:", rdkit.__version__)
print("All good!")
```

---

## 6. Get the dataset

Download `curated-solubility-dataset.csv` from Harvard Dataverse:
https://doi.org/10.7910/DVN/OVHAW8

Place it in `data/raw/`.

---

## Updating the environment

If `environment.yml` changes (new dependency added):

```bash
conda activate molgraph
conda env update -f environment.yml --prune
```
