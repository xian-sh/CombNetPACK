"""
Training and Evaluation Functions
==================================
Functions for velocity prediction model training and evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error


def compute_velocity_loss(pred_velocities, true_velocities, masks, config):
    """
    Compute velocity prediction loss
    
    Args:
        pred_velocities (Tensor): Predicted velocities, shape (B, max_atoms, 3)
        true_velocities (Tensor): True velocities, shape (B, max_atoms, 3)
        masks (Tensor): Atom masks, shape (B, max_atoms)
        config: Configuration object
    
    Returns:
        tuple: (total_loss, velocity_loss)
    """
    # Extract valid atoms
    valid_velocities_pred = pred_velocities[masks]  # (N_valid_atoms, 3)
    valid_velocities_true = true_velocities[masks]  # (N_valid_atoms, 3)
    
    # MSE loss for velocities
    velocity_loss = F.mse_loss(valid_velocities_pred, valid_velocities_true)
    
    # Total loss
    total_loss = config.VELOCITY_WEIGHT * velocity_loss
    
    return total_loss, velocity_loss


def train_epoch(model, dataloader, optimizer, config, device, epoch, logger=None):
    """
    Train for one epoch
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        optimizer (Optimizer): Optimizer
        config: Configuration object
        device (torch.device): Device to use
        epoch (int): Current epoch number
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (avg_loss, avg_velocity_loss)
    """
    model.train()
    total_loss = 0
    total_velocity_loss = 0
    
    progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
    
    for batch in progress_bar:
        features = batch['features'].to(device)
        masks = batch['masks'].to(device)
        true_velocities = batch['velocities'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred_velocities = model(features, masks)
        
        # Compute loss
        loss, velocity_loss = compute_velocity_loss(
            pred_velocities, true_velocities, masks, config
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP)
        
        optimizer.step()
        
        # Accumulate statistics
        total_loss += loss.item()
        total_velocity_loss += velocity_loss.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'V_loss': f'{velocity_loss.item():.4f}'
        })
    
    n_batches = len(dataloader)
    return total_loss / n_batches, total_velocity_loss / n_batches


@torch.no_grad()
def evaluate_model(model, dataloader, config, device, logger=None):
    """
    Evaluate model on validation/test set
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        config: Configuration object
        device (torch.device): Device to use
        logger (logging.Logger): Logger instance
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    total_loss = 0
    total_velocity_loss = 0
    
    all_velocity_pred = []
    all_velocity_true = []
    
    for batch in dataloader:
        features = batch['features'].to(device)
        masks = batch['masks'].to(device)
        true_velocities = batch['velocities'].to(device)
        
        # Forward pass
        pred_velocities = model(features, masks)
        
        # Compute loss
        loss, velocity_loss = compute_velocity_loss(
            pred_velocities, true_velocities, masks, config
        )
        
        total_loss += loss.item()
        total_velocity_loss += velocity_loss.item()
        
        # Collect predictions (only valid atoms)
        valid_velocities_pred = pred_velocities[masks].cpu().numpy()
        valid_velocities_true = true_velocities[masks].cpu().numpy()
        all_velocity_pred.extend(valid_velocities_pred)
        all_velocity_true.extend(valid_velocities_true)
    
    n_batches = len(dataloader)
    
    # Calculate metrics
    velocity_r2 = r2_score(all_velocity_true, all_velocity_pred)
    velocity_mae = mean_absolute_error(all_velocity_true, all_velocity_pred)
    
    return {
        'total_loss': total_loss / n_batches,
        'velocity_loss': total_velocity_loss / n_batches,
        'velocity_r2': velocity_r2,
        'velocity_mae': velocity_mae
    }


def save_checkpoint(model, optimizer, epoch, metrics, config, save_path, logger=None):
    """
    Save model checkpoint
    
    Args:
        model (nn.Module): Model to save
        optimizer (Optimizer): Optimizer
        epoch (int): Current epoch
        metrics (dict): Evaluation metrics
        config: Configuration object
        save_path (str): Path to save checkpoint
        logger (logging.Logger): Logger instance
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config,
    }
    
    torch.save(checkpoint, save_path)
    
    if logger:
        logger.info(f"Checkpoint saved to {save_path}")