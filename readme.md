# CombNetPack: Dynamic Atomic Representations for Combustion Modeling

**From Interatomic Potentials to Fuel Properties**

[[Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)  
[[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)  
[[License](https://img.shields.io/badge/license-CC--BY--4.0-green)](LICENSE)  
[[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17972807.svg)](https://doi.org/10.5281/zenodo.17972807)

> **Dynamic Atomic Representations: From Interatomic Potentials to Fuel Properties**  
> Zhan Si¹, Jingjing Hu², Qiqi Zhang³, Deguang Liu¹*, Haizhu Yu³*, Yao Fu¹*  
> ¹University of Science and Technology of China  
> ²Hefei University of Technology  
> ³Anhui University  
> *Corresponding authors: ldg123@mail.ustc.edu.cn; yuhaizhu@ahu.edu.cn; fuyao@ustc.edu.cn

---

![Overview](docs/imgs/fig_toc_01.png)

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Datasets](#datasets)
  - [HAC Database (Zenodo)](#hac-database-zenodo)
- [Examples](#examples)
  - [Example 1 — HAC (AIMD/Quantum Chemistry)](#example-1--hac-aimdquantum-chemistry)
  - [Example 2 — LFS (Laminar Flame Speed)](#example-2--lfs-laminar-flame-speed)
- [Citation](#citation)
- [License](#license)

---

## Overview

**CombNetPack** is a unified end-to-end deep learning framework for combustion chemistry that constructs **dynamic atomic embeddings** directly from local atomic geometry and electronic structure—**without predefined chemical bonds**.

![CombNetPack Framework](docs/imgs/fig_main_01.png)  
*Figure 1: Overall architecture of CombNetPack for unified prediction of combustion properties from atomic coordinates to fuel performance metrics.*

---

## Key Features

### Dynamic Atomic Representation (GAF)

- **GACE (Graph Atomic Cluster Expansion)**: many-body topological interactions via message-passing  
- **ACF (Atom-Centered Features)**: geometric (radial, angular) and electrostatic descriptors  

### Multi-Task Learning

| Scale | Properties | Accuracy |
|-------|-----------|----------|
| **Microscopic** | Atomic forces, energies, velocities | R² > 0.99 |
| **Thermodynamic** | ΔHc, ΔHf, HoV | R² > 0.98 |
| **Combustion** | LFS, RON, MON, ignition delay | R² > 0.95 |

---

## Architecture

```text
CombNetPack/
│
├── HAC/                            # Hydrogen–Ammonia Combustion Project
│   ├── data/
│   │   ├── raw/                    # Raw AIMD trajectories (.npz)
│   │   └── processed/              # Extracted GAF features
│   ├── ckpts/
│   ├── config.py                   # HAC configuration
│   ├── dataset.py                  # HAC dataset loader
│   ├── features.py                 # GAF feature extraction
│   ├── models.py                   # CombNet architecture
│   ├── train.py                    # Training pipeline
│   ├── run.py                      # Main entry point
│   └── run.sh                      # Execution script
│
├── LFS/                            # Laminar Flame Speed Project
│   ├── data/
│   ├── ckpts/
│   ├── config.py                   # LFS configuration
│   ├── data_processing.py          # SMILES to 3D conformer
│   ├── features.py                 # Feature extraction
│   ├── models.py                   # Multi-output CombNet
│   ├── training.py                 # Training pipeline
│   └── run.py                      # Main entry point
│
├── docs/
├── requirements.txt
└── README.md
```

---

## Installation

### Prerequisites

- **CUDA**: 11.8+
- **Python**: 3.8+
- **PyTorch**: 2.0+

### Quick Install

```bash
git clone https://github.com/yourusername/CombNetPack.git
cd CombNetPack

conda create -n combnet python=3.10
conda activate combnet

# PyTorch + CUDA 11.8 (example)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# PyTorch Geometric
conda install pyg -c pyg

# Chemistry tools
conda install -c conda-forge ase rdkit

# Common deps
pip install numpy scipy pandas matplotlib seaborn tqdm pyyaml tensorboard jupyter
```

---

## Datasets

### HAC Database (Zenodo)

**DOI:** https://doi.org/10.5281/zenodo.17972807

From the Zenodo record:

- ORCA **5.0.4**
- **B3LYP-D3(BJ)/def2-TZVP**
- Temperature: **500–3000 K**
- AIMD timestep: **0.4–1.3 fs**
- **120,000** configurations per reaction channel  
- Total **840,000** geometry–energy pairs (7 channels × 120,000)
- Files are **restricted** (record is public)

**Reaction channels (Zenodo):**
- HAC1: H₂NO + NH₂ → NH₃ + HNO (7 atoms)  
- HAC2: N₂O + H → N₂ + OH (4 atoms)  
- HAC3: NH₂ + H → NH + H₂ (4 atoms)  
- HAC4: NH₂ + HO₂ → H₂NO + OH (6 atoms)  
- HAC5: NH₂ + NO → N₂ + H₂O (5 atoms)  
- HAC6: NH₃ + OH → NH₂ + H₂O (6 atoms)  
- HAC7: NH₃ + O₂ → NH₂ + HO₂ (6 atoms)  
- HAC8: NO₂ + NH₂ → H₂NO + NO (6 atoms)

---

## Examples

### Example 1 — HAC (AIMD/Quantum Chemistry)

#### HAC Training Configuration

```python
# config.py
class TrainingConfig:
    """Configuration for velocity prediction model training"""
    DATA_DIR = "./data"
    DATA_PATTERN = "*_with_gaf.npz"

    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 1
    RANDOM_SEED = 42

    ACE_DIM = 32
    ATOM_C_DIM = 32
    TOTAL_ATOM_DIM = 64
    HIDDEN_DIM = 128
    D_ATTN = 256
    N_HEADS = 8
    N_LAYERS = 2
    DROPOUT = 0.1

    VELOCITY_DIM = 3
    VELOCITY_WEIGHT = 1.0
    VELOCITY_CONVERSION = 2625.5

    GRADIENT_CLIP = 1.0
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 10

    TEST_EVAL_FREQUENCY = 20

    SAVE_PATH = "./ckpts/best_velocity_model.pth"
    LOG_FILE = "./ckpts/velocity_training.log"
    PLOT_PATH = "./ckpts/velocity_training_curves.png"

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
```

#### Download & Run

```bash
# Download from Zenodo (requires access since files are restricted)
wget https://zenodo.org/records/17972807/files/HAC_database.zip
unzip HAC_database.zip -d HAC/data/raw/

cd HAC
bash run.sh

# Or:
python run.py extract
python run.py train
```

---

### Example 2 — LFS (Laminar Flame Speed)

#### LFS Configuration (as provided)

```python
"""
Configuration Parameters
========================
Global configuration for data processing, model architecture, and training.
"""
class Config:
    """
    Global configuration for data processing, model architecture, and training.
    """
    # ===== Data Paths =====
    INPUT_CSV_PATH = "./data/raw/LFS_pci.csv"                      # Ensure Input existence
    INPUT_NPZ_PATH = "./data/raw/lfs_dataset_3d_etkdg.npz"         # Ensure Input existence
    OUTPUT_DIR = "."
    DATASET_PATH = "./data/processed/lfs_atomic_features_uncompressed.npz"
    MODEL_SAVE_PATH = "./ckpts/best_CombNet_atomic_features_model.pth"
    LOG_FILE = "./ckpts/training.log"
    USE_COMP=False

    # ===== Dataset Split Ratios =====
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2

    # ===== Training Hyperparameters =====
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 1

    # ===== Model Architecture Dimensions =====
    D_ATOM = 64
    D_MOL = 128
    D_COND = 3
    D_ATTN = 128

    # ===== Atomic Feature Configuration =====
    ATOMIC_FEATURE_DIM = 68
    MAX_ATOMS = 150

    # ===== CombNet Specific Parameters =====
    CUTOFF = 5.0
    N_INTERACTIONS = 3
    N_RBF = 20

    # ===== GNN Feature Builder Parameters =====
    L_LIST_DIM = 10
    ATOMIC_EMB_DIM = 16
    GNN_HIDDEN_DIM = 32
    GNN_NUM_LAYERS = 3

    # ===== Atom C Descriptor Parameters =====
    ATOM_C_CUTOFF = 5.0
    N_RADIAL_BASIS = 30
    N_SPHERICAL_HARMONICS = 30
    MAX_L = 5

    # ===== CombNet Encoder Parameters =====
    CombNet_HIDDEN_DIM = 64

    # ===== Final MLP Parameters =====
    FINAL_MLP_HIDDEN1 = 256
    FINAL_MLP_HIDDEN2 = 128
    DROPOUT_RATE = 0.1
```

#### Run

```bash
cd LFS
python run.py
```

---

## Performance Benchmarks

### Atomic Force Prediction

![Performance Comparison](docs/imgs/fig_result_01.png)  
*Figure 3: Evaluation on HAC dataset and feature comparisons.*

### Fuel Properties Prediction

![Fuel Properties](docs/imgs/fig_vis_01.png)  
*Figure 4: Multi-property prediction results (including LFS, RON/MON, HoV, ΔHc/ΔHf).*

---

## Citation

```bibtex
@article{si2025combnetpack,
  title={Dynamic Atomic Representations: From Interatomic Potentials to Fuel Properties},
  author={Si, Zhan and Hu, Jingjing and Zhang, Qiqi and Liu, Deguang and Yu, Haizhu and Fu, Yao},
  journal={[Journal Name]},
  year={2025},
  note={In preparation}
}

@dataset{si2025hac,
  author={Si, Zhan and Hu, Jingjing and Liu, Deguang and Yu, Haizhu},
  title={HAC Database: Hydrogen-Ammonia Combustion AIMD Dataset},
  year={2025},
  publisher={Zenodo},
  doi={10.5281/zenodo.17972807},
  url={https://zenodo.org/records/17972807}
}
```

---

## License

- **Code**: MIT License  
- **HAC Database**: **CC BY 4.0** (Creative Commons Attribution 4.0 International)
```
