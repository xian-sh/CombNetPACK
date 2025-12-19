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
    MODEL_SAVE_PATH = "./ckpts/best_schnet_atomic_features_model.pth"
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
    
    # ===== SchNet Specific Parameters =====
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
    
    # ===== SchNet Encoder Parameters =====
    SCHNET_HIDDEN_DIM = 64
    
    # ===== Final MLP Parameters =====
    FINAL_MLP_HIDDEN1 = 256
    FINAL_MLP_HIDDEN2 = 128
    DROPOUT_RATE = 0.1