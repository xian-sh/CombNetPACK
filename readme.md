# CombNetPack: Dynamic Atomic Representations for Combustion Modeling

**From Interatomic Potentials to Fuel Properties**

[[Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[[PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[[License](https://img.shields.io/badge/license-CC--BY--4.0-green)](LICENSE)
[[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17972807.svg)](https://doi.org/10.5281/zenodo.17972807)

> **Dynamic Atomic Representations: From Interatomic Potentials to Fuel Properties**  
> Zhan Si¹, Jingjing Hu², Qiqi Zhang³, Deguang Liu¹*, Haizhu Yu³*, Yao Fu¹*  
> 
> ¹University of Science and Technology of China  
> ²Hefei University of Technology  
> ³Anhui University  
> 
> *Corresponding authors: ldg123@mail.ustc.edu.cn; yuhaizhu@ahu.edu.cn; fuyao@ustc.edu.cn

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [HAC Database](#hac-database)
- [Quick Start](#quick-start)
- [Model Training](#model-training)
- [Performance Benchmarks](#performance-benchmarks)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Overview

**CombNetPack** is a unified end-to-end deep learning framework that reimagines molecular representation for extreme reactive environments in combustion chemistry. Unlike conventional methods that rely on static molecular graphs or handcrafted features, CombNetPack constructs **dynamic atomic embeddings** directly from local atomic geometry and electronic structure—without predefined chemical bonds.

### Why CombNetPack?

Combustion is a multiscale physicochemical process spanning microscopic dynamics (femtosecond bond breaking/forming) to macroscopic flame propagation. Traditional approaches face critical limitations:

- **Feature-based ML**: Assumes persistent functional groups, fails under bond scission
- **Graph Neural Networks**: Requires fixed covalent bonding graphs, breaks down for radicals/ion pairs
- **Static Representations**: Cannot capture transient, non-bonded interactions at high temperatures (>1500 K)

CombNetPack overcomes these bottlenecks through:

1. **Bond-agnostic representation**: Constructs features from 3D atomic coordinates alone
2. **Physics-informed architecture**: Integrates Koopman dynamics and attention-enhanced continuous filters
3. **Multi-scale prediction**: Simultaneously predicts atomic forces (R² > 0.99) and macroscopic combustion properties (R² > 0.95)

![CombNetPack Framework](docs/images/framework_overview.png)
*Figure 1: Overall architecture of CombNetPack for unified prediction of combustion properties from atomic coordinates to fuel performance metrics.*

---

## 🌟 Key Features

### Dynamic Atomic Representation (GAF)

**Graph-Adaptive Atomic Features (GAF)** module constructs adaptive, physics-informed embeddings:

- **GACE (Graph Atomic Cluster Expansion)**: Many-body topological interactions via message-passing
- **ACF (Atom-Centered Features)**: Geometric (radial, angular) and electrostatic descriptors
- **Transferable**: Outperforms SOAP, Coulomb Matrix, ACSF, and Spherical Harmonics

### Multi-Task Learning

Single framework predicts:

| Scale | Properties | Accuracy |
|-------|-----------|----------|
| **Microscopic** | Atomic forces, energies, velocities | R² > 0.99 |
| **Thermodynamic** | ΔHc, ΔHf, HoV | R² > 0.98 |
| **Combustion** | LFS, RON, MON, ignition delay | R² > 0.95 |

### Robust Transferability

- **Cross-system**: Effective transfer between NH₃-H₂ and H₂-O₂ combustion systems
- **Scale-invariant**: Maintains accuracy across systems differing by 3× in atom count
- **Low-data regime**: Transfer learning reduces error by 40% with limited training data

---

## 🏗 Architecture

```
CombNetPack/
│
├── HAC/                           # Ammonia-Hydrogen Combustion Project
│   ├── data/                       
│   │   ├── raw/                   # Raw AIMD trajectories (.npz)
│   │   └── processed/             # Extracted GAF features
│   │
│   ├── ckpts/                     # Model checkpoints
│   │
│   ├── config.py                  # HAC configuration
│   ├── dataset.py                 # HAC dataset loader
│   ├── features.py                # GAF feature extraction
│   ├── models.py                  # CombNet architecture
│   ├── train.py                   # Training pipeline
│   ├── run.py                     # Main entry point
│   └── run.sh                     # Execution script
│
├── LFS/                           # Laminar Flame Speed Project
│   ├── data/                       
│   ├── ckpts/                      
│   │
│   ├── config.py                  # LFS configuration
│   ├── data_processing.py         # SMILES to 3D conformer
│   ├── features.py                # Multi-property features
│   ├── models.py                  # Multi-output CombNet
│   ├── training.py                # Multi-task training
│   └── run.py                     # Main entry point
│
├── docs/                          
├── requirements.txt               
└── README.md                      
```

### Core Components

![CombNetPack Components](docs/images/architecture_details.png)
*Figure 2: (A) GAF module architecture, (B) Attention-CFConv interaction block, (C) Koopman autoencoder for dynamics-aware representation.*

**1. Graph-Adaptive Atomic Features (GAF)**
- Treats atoms as dynamic graph nodes
- Encodes multi-body expansion via GACE + ACF
- Output: 32-128D adaptive atomic embeddings

**2. CombNet Core**
- **Interaction Blocks**: Atom-wise MLP + Attention-CFConv (5 Å cutoff)
- **Koopman Autoencoder**: Lifts features to linear-dynamics latent space
- **Multi-head Output**: Separate branches for forces, energies, velocities, and fuel properties

**3. Physics-Informed Modules**
- **Morse Potential**: Encodes bond dissociation behavior
- **Attention Mechanism**: Context-aware neighbor weighting
- **Condition Encoding**: Integrates temperature, pressure, equivalence ratio

---

## 💻 Installation

### Prerequisites

- **Hardware**: NVIDIA GPU with ≥16GB VRAM (RTX 3090/A100 recommended)
- **Software**: CUDA 11.8+, Python 3.8+

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/CombNetPack.git
cd CombNetPack

# Create conda environment
conda create -n combnet python=3.10
conda activate combnet

# Install PyTorch with CUDA 11.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install PyTorch Geometric
conda install pyg -c pyg

# Install chemistry tools
conda install -c conda-forge ase rdkit

# Install remaining dependencies
pip install numpy scipy pandas matplotlib seaborn tqdm pyyaml tensorboard jupyter
```

### Verify Installation

```python
import torch
import torch_geometric
from rdkit import Chem

print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
print(f"PyG: {torch_geometric.__version__}")
```

---

## 📊 HAC Database

### Overview

The **HAC (Hydrogen-Ammonia Combustion) Database** is a high-fidelity quantum chemical dataset specifically constructed for CombNetPack training. It contains **96,000 molecular configurations** sampled from seven critical ammonia-hydrogen combustion reaction pathways.

**DOI**: [10.5281/zenodo.17972807](https://doi.org/10.5281/zenodo.17972807)

### Computational Details

| Parameter | Value |
|-----------|-------|
| **DFT Functional** | B3LYP-D3(BJ) |
| **Basis Set** | def2-TZVP |
| **Temperature Range** | 500–3000 K |
| **Time Step** | 0.4–1.3 fs |
| **Total Configurations** | 96,000 (sampled from 840,000) |
| **Annotations** | Energy, forces, atomic positions |

### Reaction Channels

| ID | Reaction | Atoms | Description |
|----|----------|-------|-------------|
| **HAC1** | H₂NO + NH₂ → NH₃ + HNO | 7 | Hydrogen transfer |
| **HAC2** | N₂O + H → N₂ + OH | 4 | Nitrous oxide reduction |
| **HAC3** | NH₂ + H → NH + H₂ | 4 | Radical hydrogen abstraction |
| **HAC4** | NH₂ + HO₂ → H₂NO + OH | 6 | Peroxy radical reaction |
| **HAC5** | NH₂ + NO → N₂ + H₂O | 5 | DeNOₓ pathway |
| **HAC6** | NH₃ + OH → NH₂ + H₂O | 6 | Ammonia oxidation |
| **HAC7** | NH₃ + O₂ → NH₂ + HO₂ | 6 | Oxygen activation |
| **HAC8** | NO₂ + NH₂ → H₂NO + NO | 6 | Nitrogen dioxide reduction |

![HAC Reaction Pathways](docs/images/hac_reactions.png)
*Figure 3: Representative snapshots from HAC3 (hydrogen transfer) and HAC4 (oxygen transfer) reaction trajectories showing transient radical intermediates.*

### Data Structure

```
HAC_database/
├── HAC1_raw.npz          # 12,000 configs per channel
├── HAC2_raw.npz
├── ...
└── HAC8_raw.npz

# Each .npz file contains:
{
    'R': (N_configs, N_atoms, 3),      # Atomic positions (Å)
    'E': (N_configs,),                  # Total energy (Hartree)
    'F': (N_configs, N_atoms, 3),       # Atomic forces (Hartree/Bohr)
    'Z': (N_atoms,),                    # Atomic numbers
    'V': (N_configs, N_atoms, 3)        # Velocities (optional)
}
```

### Download & Usage

```bash
# Download from Zenodo
wget https://zenodo.org/records/17972807/files/HAC_database.zip
unzip HAC_database.zip -d HAC/data/raw/

# Preprocess features
cd HAC
python run.py --mode preprocess --data_path data/raw/
```

---

## 🚀 Quick Start

### 1. Train on HAC Database

```bash
cd HAC

# Full training pipeline (feature extraction + model training)
bash run.sh

# Or step-by-step:
# Step 1: Extract GAF features
python run.py --mode preprocess --config config.py

# Step 2: Train CombNet
python train.py --config config.py --epochs 500 --batch_size 32 --lr 1e-4
```

### 2. Predict Atomic Forces

```python
import torch
from models import CombNet
from dataset import HACDataset

# Load trained model
model = CombNet.load_from_checkpoint('ckpts/best_model.pt')
model.eval()

# Load test data
test_data = HACDataset('data/processed/HAC3_test.pt')
batch = test_data[0]

# Predict
with torch.no_grad():
    pred_energy, pred_forces = model(batch['pos'], batch['Z'], batch['batch'])

print(f"Energy MAE: {torch.abs(pred_energy - batch['E']).mean():.4f} Hartree")
print(f"Force MAE: {torch.abs(pred_forces - batch['F']).mean():.4f} Hartree/Bohr")
```

### 3. Predict Fuel Properties

```bash
cd LFS

# Train multi-property model
python run.py --task all --epochs 300

# Predict laminar flame speed
python run.py --task predict --smiles "CC(C)O" --temp 350 --pressure 1.0 --phi 1.0
# Output: LFS = 38.2 cm/s
```

---

## 🔬 Model Training

### HAC Training Configuration

```python
# config.py
CONFIG = {
    # Data
    'data_path': 'data/raw/',
    'split_ratio': [0.6, 0.2, 0.2],  # train/val/test
    
    # Model Architecture
    'gaf_dim': 64,
    'num_interactions': 6,
    'cutoff': 5.0,  # Angstrom
    'num_gaussians': 50,
    'koopman_dim': 128,
    
    # Training
    'epochs': 500,
    'batch_size': 32,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'grad_clip': 1.0,
    
    # Loss Weights
    'loss_weights': {
        'energy': 1.0,
        'forces': 100.0,
        'velocity': 10.0
    },
    
    # Scheduler
    'scheduler': 'ReduceLROnPlateau',
    'patience': 50,
    'factor': 0.5,
    
    # Early Stopping
    'early_stop_patience': 100
}
```

### Training Outputs

```
Epoch 100/500
Train Loss: 0.0324 | Val Loss: 0.0412
  Energy MAE: 0.0012 Hartree (0.75 kJ/mol)
  Force MAE: 0.0089 Hartree/Bohr
  Velocity MAE: 0.0145 Bohr/fs
Learning Rate: 5.0e-5
Time: 2.3 min

✓ Best model saved to ckpts/epoch_100.pt
```

### Transfer Learning

```python
# Pretrain on HAC
python train.py --config config.py --save_path ckpts/hac_pretrained.pt

# Fine-tune on new system (e.g., H2-O2)
python train.py \
    --pretrained ckpts/hac_pretrained.pt \
    --data_path data/h2o2_combustion/ \
    --freeze_gaf \
    --epochs 200
```

---

## 📈 Performance Benchmarks

### Atomic Force Prediction

![Performance Comparison](docs/images/performance_comparison.png)
*Figure 4: Comprehensive evaluation of CombNetPack on HAC dataset. (A) Comparison against GNN, GAT, SchNet, Graphormer; (B-C) Feature importance analysis for HAC3/HAC4; (D) GAF vs. SOAP/CM/ACSF/SH.*

| Dataset | Metric | CombNetPack | SchNet | Graphormer |
|---------|--------|-------------|---------|------------|
| **HAC3** | Energy R² | **0.9987** | 0.9821 | 0.9745 |
|  | Force MAE (Ha/Bohr) | **0.0078** | 0.0142 | 0.0189 |
| **HAC7** | Energy R² | **0.9912** | 0.9654 | 0.9587 |
|  | Force MAE | **0.0095** | 0.0178 | 0.0203 |
| **HC8** | Energy R² | **0.9955** | 0.9812 | 0.9768 |
| **HC15** | Energy R² | **0.9395** | 0.8987 | 0.8821 |

### Fuel Properties Prediction

![Fuel Properties](docs/images/fuel_properties.png)
*Figure 5: Multi-property prediction results on test sets. (A-B) RON/MON, (C) LFS, (D) HoV, (E-F) ΔHc/ΔHf. Representative molecular structures with predictions shown in (G-I).*

| Property | Dataset Size | R² | MAE |
|----------|--------------|-----|-----|
| **Laminar Flame Speed (LFS)** | 3,592 | **0.982** | 2.14 cm/s |
| **Heat of Combustion (ΔHc)** | 2,694 | **0.991** | 45.3 kJ/mol |
| **Enthalpy of Formation (ΔHf)** | 2,694 | **0.987** | 12.8 kJ/mol |
| **Research Octane Number (RON)** | 788 | 0.912 | 3.67 |
| **Motor Octane Number (MON)** | 480 | 0.895 | 4.12 |
| **Heat of Vaporization (HoV)** | 420 | 0.924 | 1.89 kJ/mol |

### Transferability Analysis

| System | Pretraining | Force MAE (with TL) | Force MAE (from scratch) | Improvement |
|--------|-------------|---------------------|--------------------------|-------------|
| **CMS I** (NH₃:H₂:O₂) | HAC+HC | 0.0124 Ha/Bohr | 0.0198 Ha/Bohr | **37.4%** |
| **CMS II** (H₂:O₂) | HAC+HC | 0.0089 Ha/Bohr | 0.0145 Ha/Bohr | **38.6%** |

*Note: Transfer learning evaluated with 500 training samples; improvement most pronounced in low-data regime (<1000 samples).*

### Free Energy Surface Reproduction

CombNetPack accurately reproduces temperature-dependent free energy surfaces (FES) from AIMD:

- **500 K**: Low-energy basins nearly indistinguishable from AIMD
- **1500 K**: Correct thermal population evolution
- **3000 K**: Enhanced conformational sampling while preserving dominant pathways

Cosine similarity of force vectors: **>90%** for majority of atoms across all temperatures.

---

## 📚 Citation

If you use CombNetPack or the HAC database in your research, please cite:

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

## 📄 License

- **Code**: MIT License
- **HAC Database**: Creative Commons Attribution 4.0 International (CC BY 4.0)

The HAC database allows redistribution and reuse with appropriate credit to the creators.

---

## 🙏 Acknowledgments

This work was supported by:
- National Natural Science Foundation of China (NSFC)
- Anhui Province Key Laboratory of Biomass Clean Energy
- CAS Key Laboratory of Urban Pollutant Conversion

We thank the Zenodo team for hosting the HAC database and the open-source communities behind PyTorch, PyTorch Geometric, ASE, and RDKit.

---

## 📧 Contact

For questions, issues, or collaboration inquiries:

- **Zhan Si**: [GitHub Issues](https://github.com/yourusername/CombNetPack/issues)
- **Prof. Deguang Liu**: ldg123@mail.ustc.edu.cn
- **Prof. Haizhu Yu**: yuhaizhu@ahu.edu.cn
- **Prof. Yao Fu**: fuyao@ustc.edu.cn

---

## 🔗 Related Resources

- **Paper Preprint**: [arXiv:XXXX.XXXXX] (Coming soon)
- **HAC Database**: [Zenodo Record](https://zenodo.org/records/17972807)
- **Documentation**: [Full Docs](docs/)
- **Tutorial Notebooks**: [examples/](examples/)

---

**Last Updated**: December 19, 2025  
**Version**: 1.0.0