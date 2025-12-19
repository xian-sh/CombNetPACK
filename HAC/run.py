"""
Main Execution Script
=====================
Main script for GAF feature extraction and velocity prediction
"""

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import GAFConfig, TrainingConfig
from utils import setup_logger, load_npz_file
from features import compute_descriptor, NodeFeatureBuilder
from models import SimpleGNN, VelocityTransformer
from dataset import load_velocity_dataset, create_data_loaders
from train import train_epoch, evaluate_model, save_checkpoint

from torch_geometric.utils import dense_to_sparse


# ============================================================================
# GAF Feature Extraction (from previous version)
# ============================================================================

def compute_gaf_features(R, Z, max_cutoff, device, logger=None):
    """
    Compute GAF features for molecular structures
    
    Args:
        R (np.ndarray): Atomic coordinates, shape (batch_size, n_atoms, 3)
        Z (np.ndarray): Atomic numbers, shape (batch_size, n_atoms)
        max_cutoff (float): Maximum cutoff distance
        device (str): Device to use
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (ace_features, atom_c_features)
    """
    batch_size, n_atoms, _ = R.shape
    
    # Initialize feature builder and GNN
    node_feature_builder = NodeFeatureBuilder(
        l_list_dim=GAFConfig.L_LIST_DIM,
        atomic_emb_dim=GAFConfig.ATOMIC_EMB_DIM
    ).to(device)
    
    node_dim = GAFConfig.ATOMIC_EMB_DIM + 2 + GAFConfig.L_LIST_DIM
    gnn = SimpleGNN(
        node_dim=node_dim,
        hidden_dim=GAFConfig.GNN_HIDDEN_DIM,
        num_layers=GAFConfig.GNN_NUM_LAYERS
    ).to(device)
    
    ace_features_list = []
    atom_c_features_list = []
    
    if logger:
        logger.info("Starting GAF feature computation...")
    
    for batch_idx in tqdm(range(batch_size), desc="Processing molecules"):
        coords = R[batch_idx]
        atomic_numbers = Z[batch_idx]
        
        # Filter out padding atoms
        valid_mask = atomic_numbers > 0
        if not valid_mask.any():
            ace_features_list.append(np.zeros((n_atoms, 32)))
            atom_c_features_list.append(np.zeros((n_atoms, 32)))
            continue
            
        valid_coords = coords[valid_mask]
        valid_atomic_numbers = atomic_numbers[valid_mask]
        n_valid = len(valid_coords)
        
        # Compute GNN Features
        try:
            pos_tensor = torch.tensor(valid_coords, dtype=torch.float32, device=device)
            dist_matrix = torch.cdist(pos_tensor, pos_tensor)
            
            adj_matrix = (dist_matrix <= max_cutoff) & (dist_matrix > 0)
            edge_index, _ = dense_to_sparse(adj_matrix)
            
            if edge_index.size(1) == 0:
                edge_index = torch.stack([torch.arange(n_valid), torch.arange(n_valid)]).to(device)
            
            atomic_numbers_tensor = torch.tensor(valid_atomic_numbers, dtype=torch.long, device=device)
            
            max_nu = [2.0] * n_valid
            max_l = [1.0] * n_valid
            l_list = [[1.0] * GAFConfig.L_LIST_DIM] * n_valid
            
            node_feat = node_feature_builder(atomic_numbers_tensor, max_nu, max_l, l_list, device)
            
            with torch.no_grad():
                ace_output = gnn(node_feat, pos_tensor, edge_index)
                ace_output_np = ace_output.cpu().numpy()
        
        except Exception as e:
            if logger:
                logger.error(f"GNN computation error (batch {batch_idx}): {e}")
            ace_output_np = np.zeros((n_valid, 32))
        
        # Compute ACE-like Descriptors
        atom_c_output = np.zeros((n_valid, 32))
        
        for atom_idx in range(n_valid):
            try:
                descriptor = compute_descriptor(
                    center_coords=valid_coords[atom_idx],
                    all_coords=valid_coords,
                    all_types=valid_atomic_numbers,
                    center_type=valid_atomic_numbers[atom_idx],
                    cutoff=max_cutoff
                )
                atom_c_output[atom_idx] = descriptor
            except Exception as e:
                if logger:
                    logger.error(f"ACE descriptor error (batch {batch_idx}, atom {atom_idx}): {e}")
                atom_c_output[atom_idx] = np.zeros(32)
        
        # Pad to original size
        full_ace_features = np.zeros((n_atoms, 32))
        full_atom_c_features = np.zeros((n_atoms, 32))
        
        full_ace_features[valid_mask] = ace_output_np
        full_atom_c_features[valid_mask] = atom_c_output
        
        ace_features_list.append(full_ace_features)
        atom_c_features_list.append(full_atom_c_features)
    
    ace_features = np.array(ace_features_list)
    atom_c_features = np.array(atom_c_features_list)
    
    return ace_features, atom_c_features


def process_single_file(data_path, device, logger):
    """
    Process a single NPZ file to add GAF features
    
    Args:
        data_path (str): Path to NPZ file
        device (str): Device to use
        logger (logging.Logger): Logger instance
    """
    logger.info("=" * 60)
    logger.info(f"Processing file: {data_path}")
    logger.info("=" * 60)
    
    try:
        data = load_npz_file(data_path, logger)
        
        R = data['R']
        Z = data['Z']
        N = data['N']
        E = data['E']
        F = data['F']
        V = data['V']
        RXN = data['RXN']
        
        logger.info(f"Data shapes:")
        logger.info(f"  R: {R.shape}")
        logger.info(f"  Z: {Z.shape}")
        logger.info(f"  V: {V.shape}")
        
        # Compute features
        ace_features, atom_c_features = compute_gaf_features(
            R, Z, 
            max_cutoff=GAFConfig.MAX_CUTOFF, 
            device=device, 
            logger=logger
        )
        
        logger.info(f"Computation completed:")
        logger.info(f"  ACE features: {ace_features.shape}")
        logger.info(f"  Atom C features: {atom_c_features.shape}")
        
        # Save new NPZ file
        output_path = data_path.replace('.npz', GAFConfig.OUTPUT_SUFFIX)
        np.savez(output_path,
                 R=R, Z=Z, N=N, E=E, F=F, V=V, RXN=RXN,
                 ACE_features=ace_features,
                 Atom_C_features=atom_c_features
        )
        
        logger.info(f"Saved to: {output_path}")
        logger.info(f"File {os.path.basename(data_path)} processing completed!")
        
    except Exception as e:
        logger.error(f"Error processing file {data_path}: {e}")


def extract_gaf_features():
    """
    Main function for GAF feature extraction
    """
    logger = setup_logger("./ckpts/gaf_extraction.log")
    
    device = GAFConfig.DEVICE if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    logger.info(f"Total files to process: {len(GAFConfig.INPUT_FILES)}")
    
    # Check file existence
    existing_files = []
    for file_path in GAFConfig.INPUT_FILES:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            logger.warning(f"File not found: {file_path}")
    
    logger.info(f"Found {len(existing_files)} valid files")
    
    # Process files
    for i, data_path in enumerate(existing_files, 1):
        logger.info(f"\nProgress: [{i}/{len(existing_files)}]")
        process_single_file(data_path, device, logger)
    
    logger.info("=" * 60)
    logger.info("All files processing completed!")
    logger.info("=" * 60)


# ============================================================================
# Velocity Prediction Model Training
# ============================================================================

def plot_training_curves(train_losses, val_losses, save_path, logger=None):
    """
    Plot and save training curves
    """
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Velocity Prediction Training Curves')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(np.log10(train_losses), label='Log Training Loss')
    plt.plot(np.log10(val_losses), label='Log Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.legend()
    plt.title('Log Training Curves')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if logger:
        logger.info(f"Training curves saved to {save_path}")
    
    plt.close()


def train_velocity_model():
    """
    Main function for velocity prediction model training
    """
    logger = setup_logger(TrainingConfig.LOG_FILE)
    
    # Set random seeds
    torch.manual_seed(TrainingConfig.RANDOM_SEED)
    np.random.seed(TrainingConfig.RANDOM_SEED)
    
    # Set device
    device = torch.device(TrainingConfig.DEVICE if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    logger.info(f"Velocity unit conversion factor: {TrainingConfig.VELOCITY_CONVERSION}")
    
    # Load data
    logger.info("Loading velocity dataset...")
    dataset = load_velocity_dataset(TrainingConfig, logger)
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(dataset, TrainingConfig, logger)
    
    # Create model
    model = VelocityTransformer(TrainingConfig).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total model parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=TrainingConfig.LEARNING_RATE, 
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=TrainingConfig.LR_SCHEDULER_FACTOR, 
        patience=TrainingConfig.LR_SCHEDULER_PATIENCE, 
        verbose=True
    )
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    logger.info("\nStarting velocity prediction training...")
    
    for epoch in range(1, TrainingConfig.EPOCHS + 1):
        # Train
        train_loss, train_velocity_loss = train_epoch(
            model, train_loader, optimizer, TrainingConfig, device, epoch, logger
        )
        
        # Validate
        val_metrics = evaluate_model(
            model, val_loader, TrainingConfig, device, logger
        )
        
        # Update learning rate
        scheduler.step(val_metrics['total_loss'])
        
        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_metrics['total_loss'])
        
        # Log results
        logger.info(f"Epoch {epoch}/{TrainingConfig.EPOCHS}:")
        logger.info(f"  Train - Total: {train_loss:.6f}, Velocity: {train_velocity_loss:.6f}")
        logger.info(f"  Val   - Total: {val_metrics['total_loss']:.6f}, R²: {val_metrics['velocity_r2']:.4f}, MAE: {val_metrics['velocity_mae']:.4f}")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                TrainingConfig, TrainingConfig.SAVE_PATH, logger
            )
            logger.info("  --> Model saved (best validation loss)")
        
        # Test evaluation
        if epoch % TrainingConfig.TEST_EVAL_FREQUENCY == 0:
            test_metrics = evaluate_model(
                model, test_loader, TrainingConfig, device, logger
            )
            logger.info(f"  Test  - R²: {test_metrics['velocity_r2']:.4f}, MAE: {test_metrics['velocity_mae']:.4f}")
    
    # Final test
    logger.info("\nTraining completed! Performing final test...")
    
    checkpoint = torch.load(TrainingConfig.SAVE_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate_model(
        model, test_loader, TrainingConfig, device, logger
    )
    
    logger.info("Final test results:")
    logger.info(f"  Velocity R²: {test_metrics['velocity_r2']:.4f}")
    logger.info(f"  Velocity MAE: {test_metrics['velocity_mae']:.4f}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, TrainingConfig.PLOT_PATH, logger)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = "both"
    
    if mode == "extract" or mode == "both":
        print("\n" + "="*60)
        print("GAF FEATURE EXTRACTION")
        print("="*60 + "\n")
        extract_gaf_features()
    
    if mode == "train" or mode == "both":
        print("\n" + "="*60)
        print("VELOCITY PREDICTION MODEL TRAINING")
        print("="*60 + "\n")
        train_velocity_model()
    
    if mode not in ["extract", "train", "both"]:
        print("Usage: python main.py [extract|train|both]")
        print("  extract - Only extract GAF features")
        print("  train   - Only train velocity prediction model")
        print("  both    - Extract features then train model (default)")