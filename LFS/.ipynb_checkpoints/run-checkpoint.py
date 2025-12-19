"""
Main Execution Script
=====================
Entry point for the training pipeline.
"""

import os
import warnings
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from config import Config
from utils import setup_logger
from data_processing import (
    process_npz_file_for_features,
    LFS_FeatureDataset,
    collate_fn_new
)
from models import SchNetWithAttention
from training import train, evaluate


def main():
    """
    Main function for training SchNet model.
    """
    # Setup logger
    logger = setup_logger(Config.LOG_FILE)
    
    logger.info("=" * 60)
    logger.info("Training SchNet Model with 68-Dimensional Atomic Features")
    logger.info("=" * 60)
    
    if not os.path.exists(Config.DATASET_PATH):
        logger.error(f"Error: Dataset file not found at {Config.DATASET_PATH}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load and split data
    full_dataset = LFS_FeatureDataset(Config.DATASET_PATH)
    total_size = len(full_dataset)
    test_size = int(Config.TEST_SIZE * total_size)
    val_size = int(Config.VAL_SIZE * total_size)

    train_indices, temp_indices = train_test_split(
        np.arange(total_size), test_size=(test_size + val_size), random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=test_size / (test_size + val_size), random_state=42
    )

    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    val_subset = torch.utils.data.Subset(full_dataset, val_indices)
    test_subset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = DataLoader(train_subset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn_new)
    val_loader = DataLoader(val_subset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_new)
    test_loader = DataLoader(test_subset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn_new)
    
    logger.info(f"Data split: train={len(train_subset)}, val={len(val_subset)}, test={len(test_subset)}")

    # Model, optimizer, and loss function
    model = SchNetWithAttention(Config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.MSELoss() 
    
    logger.info(f"Total model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    # Training loop
    best_val_loss = float('inf')
    
    logger.info("\nStarting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train(model, train_loader, optimizer, criterion, device, logger)
        val_loss, val_r2 = evaluate(model, val_loader, criterion, device, logger)
        
        logger.info(f"Epoch {epoch}/{Config.EPOCHS}: "
                   f"Train MSE = {train_loss:.4f} | "
                   f"Val MSE = {val_loss:.4f}, Val R² = {val_r2:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"  --> Validation loss improved, model saved to {Config.MODEL_SAVE_PATH}")
        
        if epoch % 10 == 0:
            test_loss_interim, test_r2_interim = evaluate(model, test_loader, criterion, device, logger)
            logger.info("-" * 60)
            logger.info(f"  ** Epoch {epoch} Interim Test **: Test MSE = {test_loss_interim:.4f}, Test R² = {test_r2_interim:.4f}")
            logger.info("-" * 60)

    # Final test
    logger.info("\nTraining complete, loading best model for final test...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    final_test_loss, final_test_r2 = evaluate(model, test_loader, criterion, device, logger)
    
    logger.info("\n" + "=" * 20 + " Final Test Results " + "=" * 20)
    logger.info(f"  Test MSE  = {final_test_loss:.4f}")
    logger.info(f"  Test RMSE = {np.sqrt(final_test_loss):.4f}")
    logger.info(f"  Test R²   = {final_test_r2:.4f}")
    logger.info("=" * 55)


if __name__ == "__main__":
    # Setup logger
    logger = setup_logger(Config.LOG_FILE)
    
    # Step 1: Process NPZ file to extract atomic features
    logger.info("Step 1: Processing NPZ file to extract atomic features...")
    process_npz_file_for_features(
        input_npz_path=Config.INPUT_NPZ_PATH,
        output_npz_path=Config.DATASET_PATH,
        use_compression=Config.USE_COMP,
        logger=logger
    )
    
    # Step 2: Verify the generated file
    logger.info("\nStep 2: Verifying generated NPZ file...")
    try:
        data = np.load(Config.DATASET_PATH, allow_pickle=True)
        logger.info("File loaded successfully!")
        logger.info(f"Keys: {list(data.keys())}")
        logger.info(f"Number of molecules: {len(data['molecular_features'])}")
        logger.info(f"First molecule feature shape: {data['molecular_features'][0].shape}")
        logger.info(f"Conditions shape: {data['conditions'].shape}")
        logger.info(f"Targets shape: {data['targets'].shape}")
        logger.info(f"SMILES count: {len(data['smiles'])}")
        logger.info("File content is valid!")
    except Exception as e:
        logger.error(f"Error loading file: {e}")
        logger.error("You may need to regenerate the NPZ file")
        exit(1)
    
    # Step 3: Train the model
    logger.info("\nStep 3: Training SchNet model...")
    main()