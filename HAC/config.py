"""
Configuration Parameters
========================
All configuration parameters for GAF feature extraction and velocity prediction
"""

import torch


class GAFConfig:
    """Configuration for GAF feature extraction"""
    # File paths
    INPUT_FILES = [
        "./data/N2O+H-D.npz",
        "./data/H2NO+NH2-T.npz"
    ]
    OUTPUT_SUFFIX = "_with_gaf.npz"
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Feature parameters
    MAX_CUTOFF = 6.0
    N_MAX = 10
    L_MAX = 4
    
    # GNN parameters
    L_LIST_DIM = 10
    ATOMIC_EMB_DIM = 16
    GNN_HIDDEN_DIM = 32
    GNN_NUM_LAYERS = 3
    
    # ACE descriptor parameters
    ACE_OUTPUT_DIM = 32
    
    # Atomic properties (for ACE descriptor)
    ATOMIC_CHARGES = {
        1: 1.0,   # H
        6: 4.0,   # C
        7: 5.0,   # N
        8: 6.0,   # O
        9: 7.0,   # F
        15: 5.0,  # P
        16: 6.0,  # S
        17: 7.0,  # Cl
        35: 7.0,  # Br
        53: 7.0,  # I
    }
    
    ATOMIC_MASSES = {
        1: 1.008,    # H
        6: 12.011,   # C
        7: 14.007,   # N
        8: 15.999,   # O
        9: 18.998,   # F
        15: 30.974,  # P
        16: 32.065,  # S
        17: 35.453,  # Cl
        35: 79.904,  # Br
        53: 126.904, # I
    }
    
    ATOMIC_RADII = {
        1: 0.31,   # H
        6: 0.76,   # C
        7: 0.71,   # N
        8: 0.66,   # O
        9: 0.57,   # F
        15: 1.07,  # P
        16: 1.05,  # S
        17: 1.02,  # Cl
        35: 1.20,  # Br
        53: 1.39,  # I
    }



class TrainingConfig:
    """Configuration for velocity prediction model training"""
    # Data paths
    DATA_DIR = "./data"
    DATA_PATTERN = "*_with_gaf.npz"
    
    # Dataset split
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    # Training hyperparameters
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    EPOCHS = 1
    RANDOM_SEED = 42
    
    # Model architecture
    ACE_DIM = 32              # ACE features dimension
    ATOM_C_DIM = 32           # Atom_C features dimension
    TOTAL_ATOM_DIM = 64       # Total atom feature dimension (32 + 32)
    HIDDEN_DIM = 128          # Hidden layer dimension
    D_ATTN = 256              # Attention dimension
    N_HEADS = 8               # Number of attention heads
    N_LAYERS = 2              # Number of transformer layers
    DROPOUT = 0.1             # Dropout rate
    
    # Output dimensions
    VELOCITY_DIM = 3          # Velocity vector per atom (x, y, z)
    
    # Loss weights
    VELOCITY_WEIGHT = 1.0
    
    # Unit conversion constants
    VELOCITY_CONVERSION = 2625.5  # Velocity unit conversion (same as force)
    
    # Training settings
    GRADIENT_CLIP = 1.0
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 10
    
    # Evaluation frequency
    TEST_EVAL_FREQUENCY = 20
    
    # Checkpoints
    SAVE_PATH = "./ckpts/best_velocity_model.pth"
    LOG_FILE = "./ckpts/velocity_training.log"
    PLOT_PATH = "./ckpts/velocity_training_curves.png"
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'