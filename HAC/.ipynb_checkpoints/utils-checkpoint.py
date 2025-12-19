"""
Utility Functions
=================
Basic utility functions for logging, distance calculation, coordinate transformation, etc.
"""

import numpy as np
import torch
import logging
import sys
from datetime import datetime


def setup_logger(log_file):
    """
    Set up logger to write to both console and file.
    
    Args:
        log_file (str): Path to log file
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger('GAF_Training')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def compute_distance(atom1, atom2):
    """
    Calculate Euclidean distance between two atoms.
    
    Args:
        atom1 (np.ndarray): First atom coordinates, shape (3,)
        atom2 (np.ndarray): Second atom coordinates, shape (3,)
    
    Returns:
        float: Distance between atoms
    """
    return np.linalg.norm(atom1 - atom2)


def cartesian_to_spherical(atom, center):
    """
    Convert Cartesian coordinates to spherical coordinates.
    
    Args:
        atom (np.ndarray): Atom position, shape (3,)
        center (np.ndarray): Center position, shape (3,)
    
    Returns:
        tuple: (r, theta, phi) - radial distance, polar angle, azimuthal angle
    """
    diff = atom - center
    r = np.linalg.norm(diff)
    
    if r == 0:
        return 0, 0, 0
    
    theta = np.arccos(np.clip(diff[2] / r, -1, 1))  # Polar angle with clipping
    phi = np.arctan2(diff[1], diff[0])  # Azimuthal angle
    
    return r, theta, phi


def morse_potential_torch(r, D=1.0, a=1.0, r0=1.0):
    """
    Compute Morse potential: D * (1 - exp(-a*(r - r0)))^2
    
    Args:
        r (Tensor): Distances, shape (E, 1) or (E,)
        D (float): Potential well depth
        a (float): Width parameter
        r0 (float): Equilibrium distance
    
    Returns:
        Tensor: Morse potential values
    """
    return D * (1 - torch.exp(-a * (r - r0))) ** 2


def build_edge_attr(pos, row, col):
    """
    Build edge attributes: distance + Morse potential
    
    Args:
        pos (Tensor): Atomic coordinates, shape (N, 3)
        row (Tensor): Source node indices, shape (E,)
        col (Tensor): Target node indices, shape (E,)
    
    Returns:
        Tensor: Edge attributes, shape (E, 2) - [distance, morse_potential]
    """
    edge_vec = pos[row] - pos[col]  # (E, 3)
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)  # (E, 1)
    edge_morse = morse_potential_torch(edge_length)  # (E, 1)
    edge_attr = torch.cat([edge_length, edge_morse], dim=1)  # (E, 2)
    return edge_attr


def load_npz_file(file_path, logger=None):
    """
    Load and inspect NPZ file contents.
    
    Args:
        file_path (str): Path to NPZ file
        logger (logging.Logger): Logger instance
    
    Returns:
        dict: Loaded data
    """
    data = np.load(file_path, allow_pickle=True)
    
    if logger:
        logger.info(f"Keys contained: {list(data.keys())}")
        
        for key in data.keys():
            arr = data[key]
            try:
                shape = arr.shape
            except Exception:
                shape = "unknown"
            
            logger.info(f"\nKey: {key}")
            logger.info(f"  Shape: {shape}")
            logger.info(f"  Dtype: {getattr(arr, 'dtype', type(arr))}")
            
            try:
                sample = arr[:3]
            except Exception:
                sample = arr
            
            logger.info(f"  First 3 samples:\n{sample}")
    
    return data