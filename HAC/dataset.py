"""
Dataset and Data Processing
============================
Dataset classes and collate functions for molecular velocity prediction
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import os
import glob
from tqdm import tqdm
from config import TrainingConfig


class VelocityDataset(Dataset):
    """
    Dataset for molecular structures with velocities
    """
    def __init__(self, file_paths, logger=None):
        """
        Initialize dataset
        
        Args:
            file_paths (list): List of NPZ file paths with GAF features
            logger (logging.Logger): Logger instance
        """
        self.data = []
        
        if logger:
            logger.info("Loading velocity dataset...")
        
        for file_path in tqdm(file_paths, desc="Loading files"):
            try:
                data = np.load(file_path)
                
                # Extract required data
                R = data['R']                          # (N, n_atoms, 3) coordinates
                Z = data['Z']                          # (N, n_atoms) atomic types
                V = data['V']                          # (N, n_atoms, 3) velocities
                ACE_features = data['ACE_features']    # (N, n_atoms, 32)
                Atom_C_features = data['Atom_C_features']  # (N, n_atoms, 32)
                
                # Combine features
                combined_features = np.concatenate(
                    [ACE_features, Atom_C_features], 
                    axis=-1
                )  # (N, n_atoms, 64)
                
                for i in range(len(R)):
                    # Filter valid atoms (atomic type > 0)
                    valid_mask = Z[i] > 0
                    if not valid_mask.any():
                        continue
                    
                    self.data.append({
                        'features': combined_features[i][valid_mask].astype(np.float32),
                        'positions': R[i][valid_mask].astype(np.float32),
                        'atomic_numbers': Z[i][valid_mask].astype(np.int64),
                        'velocities': V[i][valid_mask].astype(np.float32),
                    })
                
                if logger:
                    logger.info(f"Loaded {len(R)} molecules from {os.path.basename(file_path)}")
                
            except Exception as e:
                if logger:
                    logger.error(f"Error loading file {file_path}: {e}")
                else:
                    print(f"Error loading file {file_path}: {e}")
        
        if logger:
            logger.info(f"Total loaded molecules: {len(self.data)}")
            if self.data:
                sample = self.data[0]
                logger.info(f"Sample structure: {list(sample.keys())}")
                logger.info(f"Feature shape: {sample['features'].shape}")
                logger.info(f"Velocity shape: {sample['velocities'].shape}")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Get a single sample
        
        Args:
            idx (int): Sample index
        
        Returns:
            dict: Sample containing features, positions, atomic_numbers, velocities
        """
        item = self.data[idx].copy()
        
        # Velocity unit conversion
        item['velocities'] = item['velocities'] * TrainingConfig.VELOCITY_CONVERSION
        
        return item


def collate_fn(batch):
    """
    Custom collate function for batching variable-length sequences
    
    Args:
        batch (list): List of samples from dataset
    
    Returns:
        dict: Batched data with padding
    """
    max_atoms = max(item['features'].shape[0] for item in batch)
    batch_size = len(batch)
    feature_dim = batch[0]['features'].shape[1]
    
    # Create padded tensors
    features = torch.zeros(batch_size, max_atoms, feature_dim)
    positions = torch.zeros(batch_size, max_atoms, 3)
    atomic_numbers = torch.zeros(batch_size, max_atoms, dtype=torch.long)
    velocities = torch.zeros(batch_size, max_atoms, 3)
    masks = torch.zeros(batch_size, max_atoms, dtype=torch.bool)
    
    for i, item in enumerate(batch):
        n_atoms = item['features'].shape[0]
        
        features[i, :n_atoms] = torch.from_numpy(item['features'])
        positions[i, :n_atoms] = torch.from_numpy(item['positions'])
        atomic_numbers[i, :n_atoms] = torch.from_numpy(item['atomic_numbers'])
        velocities[i, :n_atoms] = torch.from_numpy(item['velocities'])
        masks[i, :n_atoms] = True
    
    return {
        'features': features,
        'positions': positions,
        'atomic_numbers': atomic_numbers,
        'velocities': velocities,
        'masks': masks
    }


def create_data_loaders(dataset, config, logger=None):
    """
    Create train/validation/test data loaders
    
    Args:
        dataset (Dataset): Complete dataset
        config: Configuration object
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(config.TRAIN_RATIO * total_size)
    val_size = int(config.VAL_RATIO * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    if logger:
        logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader, test_loader


def load_velocity_dataset(config, logger=None):
    """
    Load velocity dataset from processed GAF files
    
    Args:
        config: Configuration object
        logger (logging.Logger): Logger instance
    
    Returns:
        VelocityDataset: Loaded dataset
    """
    # Find data files
    data_files = glob.glob(os.path.join(config.DATA_DIR, config.DATA_PATTERN))
    
    if logger:
        logger.info(f"Found {len(data_files)} data files:")
        for f in data_files:
            logger.info(f"  - {os.path.basename(f)}")
    
    if len(data_files) == 0:
        error_msg = "No data files found! Please run GAF feature extraction first."
        if logger:
            logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Load dataset
    dataset = VelocityDataset(data_files, logger)
    
    return dataset