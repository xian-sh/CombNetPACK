"""
Feature Computation Functions
==============================
Functions for computing atomic features, including:
- Radial basis functions
- Spherical harmonics
- Coulomb forces
- Atom C descriptors
"""

import numpy as np
import torch
import torch.nn as nn
from utils import compute_distance, cartesian_to_spherical


def compute_radial_basis_function(d, beta):
    """
    Compute radial basis function: exp(-beta * d^2).
    
    Args:
        d (float): Distance
        beta (np.ndarray): Beta parameters, shape (N,)
    
    Returns:
        np.ndarray: RBF values, shape (N,)
    """
    return np.exp(-beta * (d ** 2))


def compute_spherical_harmonics(theta, phi, l_max=5):
    """
    Simplified spherical harmonics (not standard implementation, for directional features).
    
    Args:
        theta (float): Polar angle
        phi (float): Azimuthal angle
        l_max (int): Maximum angular momentum
    
    Returns:
        np.ndarray: Spherical harmonic features, shape (30,)
    """
    features = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            if m == 0:
                features.append(np.cos(l * theta))
            elif m > 0:
                features.append(np.cos(m * phi) * np.sin(l * theta))
            else:
                features.append(np.sin(abs(m) * phi) * np.sin(l * theta))
    
    if len(features) > 30:
        return np.array(features[:30])
    else:
        result = np.zeros(30)
        result[:len(features)] = features
        return result


def compute_coulomb_force(q1, q2, r):
    """
    Compute simplified Coulomb force: q1 * q2 / r^2.
    
    Args:
        q1 (float): Charge of first atom
        q2 (float): Charge of second atom
        r (float): Distance between atoms
    
    Returns:
        float: Coulomb force magnitude
    """
    if r == 0:
        return 0.0
    return q1 * q2 / (r ** 2)


def compute_descriptor(center_coords, all_coords, all_types, center_type, cutoff=5.0):
    """
    Compute Atom C descriptor for a center atom.
    
    Descriptor includes:
    - Radial features (A): Sum of RBF over neighboring atoms
    - Directional features (B): Sum of spherical harmonics
    - Connectivity degree: Number of neighbors within cutoff
    - Total Coulomb force: Sum of Coulomb forces from neighbors
    
    Args:
        center_coords (np.ndarray): Center atom coordinates, shape (3,)
        all_coords (np.ndarray): All atom coordinates, shape (N, 3)
        all_types (list): Atom types (element symbols)
        center_type (str): Center atom type
        cutoff (float): Distance cutoff for neighbors
    
    Returns:
        np.ndarray: Descriptor vector, shape (32,) - [C(30), connectivity_degree(1), total_coulomb(1)]
    """
    charges = {
        "H": 1.0, "C": 2.0, "N": -3.0, "O": -2.0, "S": -2.0, 
        "F": -1.0, "Cl": -1.0, "Br": -1.0, "I": -1.0
    }
    
    A = np.zeros(30)
    B = np.zeros(30)
    connectivity_degree = 0
    total_coulomb_force = 0.0
    
    beta = np.linspace(0.1, 3.0, 30)
    center_charge = charges.get(center_type, 0.0)
    
    for atom_coords, atom_type in zip(all_coords, all_types):
        d = compute_distance(center_coords, atom_coords)
        
        if 0 < d <= cutoff:
            connectivity_degree += 1
            A += compute_radial_basis_function(d, beta)
            
            _, theta, phi = cartesian_to_spherical(atom_coords, center_coords)
            B += compute_spherical_harmonics(theta, phi, l_max=5)
            
            atom_charge = charges.get(atom_type, 0.0)
            total_coulomb_force += compute_coulomb_force(center_charge, atom_charge, d)
    
    C = A + B
    descriptor = np.concatenate([
        C[:30],
        [connectivity_degree],
        [total_coulomb_force]
    ])
    
    return descriptor


class NodeFeatureBuilder(nn.Module):
    """
    Build initial node features using atomic embeddings and quantum numbers.
    """
    def __init__(self, l_list_dim=10, atomic_emb_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(10, atomic_emb_dim)
        self.atomic_emb_dim = atomic_emb_dim
        self.l_list_dim = l_list_dim
        
    def forward(self, atomic_numbers, max_nu, max_l, l_list, device):
        """
        Build node features from atomic information.
        
        Args:
            atomic_numbers (list): Atomic numbers, length N
            max_nu (list): Maximum principal quantum numbers, length N
            max_l (list): Maximum angular momentum quantum numbers, length N
            l_list (np.ndarray): Angular momentum features, shape (N, l_list_dim)
            device (torch.device): Device to use
        
        Returns:
            Tensor: Node features, shape (N, atomic_emb_dim + 1 + 1 + l_list_dim)
        """
        atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long, device=device)
        max_nu = torch.tensor(max_nu, dtype=torch.float32, device=device).unsqueeze(1)
        max_l = torch.tensor(max_l, dtype=torch.float32, device=device).unsqueeze(1)
        l_list = torch.tensor(l_list, dtype=torch.float32, device=device)
        
        atomic_embeds = self.embedding(atomic_numbers)
        
        node_feat = torch.cat([
            atomic_embeds,
            max_nu,
            max_l,
            l_list
        ], dim=1)
        
        return node_feat