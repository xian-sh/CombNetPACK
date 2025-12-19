"""
Utility Functions
=================
Basic utility functions for distance calculation, coordinate transformation, etc.
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
    logger = logging.getLogger('LFS_Training')
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


def morse_potential_torch(r, D=1.0, a=1.0, r0=1.0):
    """
    Compute Morse potential for atom pairs.
    
    Args:
        r (Tensor): Distances between atoms, shape (E,) or (E, 1)
        D (float): Depth of potential well
        a (float): Width parameter
        r0 (float): Equilibrium distance
    
    Returns:
        Tensor: Morse potential values, shape (E,)
    """
    return D * (1 - torch.exp(-a * (r - r0))) ** 2


def build_edge_attr(pos, row, col):
    """
    Build edge attributes (distance and Morse potential).
    
    Args:
        pos (Tensor): Atomic coordinates, shape (N, 3)
        row (Tensor): Source node indices, shape (E,)
        col (Tensor): Target node indices, shape (E,)
    
    Returns:
        Tensor: Edge attributes, shape (E, 2) - [distance, morse_potential]
    """
    edge_vec = pos[row] - pos[col]
    edge_length = torch.norm(edge_vec, dim=1, keepdim=True)
    edge_morse = morse_potential_torch(edge_length)
    edge_attr = torch.cat([edge_length, edge_morse], dim=1)
    return edge_attr


def compute_distance(atom1, atom2):
    """
    Calculate Euclidean distance between two atoms.
    
    Args:
        atom1 (np.ndarray): Coordinates of first atom, shape (3,)
        atom2 (np.ndarray): Coordinates of second atom, shape (3,)
    
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
    vec = atom - center
    r = np.linalg.norm(vec)
    if r == 0:
        return 0.0, 0.0, 0.0
    
    theta = np.arccos(vec[2] / r)
    phi = np.arctan2(vec[1], vec[0])
    
    return r, theta, phi