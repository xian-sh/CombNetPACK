"""
Training and Evaluation Functions
==================================
Functions for training and evaluating the model.
"""

import torch
import torch.nn as nn
from sklearn.metrics import r2_score
import numpy as np


def train(model, dataloader, optimizer, criterion, device, logger=None):
    """
    Train the model for one epoch.
    
    Args:
        model (nn.Module): Model to train
        dataloader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        logger (logging.Logger): Logger instance
    
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    for atomic_features, mask, conditions, targets in dataloader:
        atomic_features, mask, conditions, targets = (
            atomic_features.to(device), mask.to(device), 
            conditions.to(device), targets.to(device)
        )
        
        optimizer.zero_grad()
        outputs = model(atomic_features, mask, conditions)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * atomic_features.size(0)
    
    return total_loss / len(dataloader.dataset)


@torch.no_grad()
def evaluate(model, dataloader, criterion, device, logger=None):
    """
    Evaluate the model.
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Evaluation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to use
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (avg_loss, r2_score)
    """
    model.eval()
    total_loss = 0
    all_targets = []
    all_outputs = []
    
    for atomic_features, mask, conditions, targets in dataloader:
        atomic_features, mask, conditions, targets = (
            atomic_features.to(device), mask.to(device), 
            conditions.to(device), targets.to(device)
        )
        
        outputs = model(atomic_features, mask, conditions)
        loss = criterion(outputs, targets)
        
        total_loss += loss.item() * atomic_features.size(0)
        
        all_targets.extend(targets.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())
    
    r2 = r2_score(np.array(all_targets), np.array(all_outputs))
    avg_loss = total_loss / len(dataloader.dataset)
    
    return avg_loss, r2