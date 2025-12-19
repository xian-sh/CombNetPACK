"""
Feature Computation
===================
Functions for computing RBF, spherical harmonics, and molecular descriptors
"""

import numpy as np
import torch
from scipy.special import sph_harm
from config import GAFConfig


# ============================================================================
# Radial Basis Functions
# ============================================================================

def gaussian_rbf(distances, n_max=10, cutoff=6.0):
    """
    Compute Gaussian Radial Basis Functions
    
    Args:
        distances (np.ndarray): Distances, shape (n,)
        n_max (int): Maximum number of basis functions
        cutoff (float): Cutoff distance
    
    Returns:
        np.ndarray: RBF values, shape (n, n_max)
    """
    n = np.arange(1, n_max + 1)
    sigma = cutoff / n_max
    centers = n * cutoff / n_max
    
    distances = distances[:, np.newaxis]  # (n_dists, 1)
    centers = centers[np.newaxis, :]      # (1, n_max)
    
    rbf = np.exp(-((distances - centers) ** 2) / (2 * sigma ** 2))
    
    # Apply cutoff function
    cutoff_mask = distances.flatten() < cutoff
    rbf[~cutoff_mask] = 0
    
    return rbf


def smooth_cutoff(r, cutoff):
    """
    Smooth cutoff function
    
    Args:
        r (float or np.ndarray): Distance(s)
        cutoff (float): Cutoff distance
    
    Returns:
        float or np.ndarray: Cutoff function value(s)
    """
    x = r / cutoff
    mask = r < cutoff
    result = np.zeros_like(r)
    result[mask] = 1 - 6 * x[mask]**5 + 15 * x[mask]**4 - 10 * x[mask]**3
    return result


# ============================================================================
# Spherical Harmonics
# ============================================================================

def compute_spherical_harmonics(vectors, l_max=4):
    """
    Compute spherical harmonics for a set of vectors
    
    Args:
        vectors (np.ndarray): 3D vectors, shape (n, 3)
        l_max (int): Maximum angular momentum
    
    Returns:
        np.ndarray: Spherical harmonics, shape (n, (l_max+1)^2)
    """
    if len(vectors) == 0:
        return np.array([])
    
    # Convert to spherical coordinates
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Avoid division by zero
    r_safe = np.where(r > 1e-10, r, 1e-10)
    
    theta = np.arccos(np.clip(z / r_safe, -1, 1))  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle
    
    # Compute spherical harmonics
    ylm_list = []
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            ylm = sph_harm(m, l, phi, theta)
            ylm_real = ylm.real
            ylm_list.append(ylm_real)
    
    ylm_array = np.stack(ylm_list, axis=1)  # (n, (l_max+1)^2)
    
    return ylm_array


# ============================================================================
# ACE-like Descriptors
# ============================================================================

def compute_descriptor(center_coords, all_coords, all_types, center_type, cutoff=6.0):
    """
    Compute ACE-like atomic descriptor for a single atom
    
    Args:
        center_coords (np.ndarray): Coordinates of center atom, shape (3,)
        all_coords (np.ndarray): Coordinates of all atoms, shape (n_atoms, 3)
        all_types (np.ndarray): Atomic numbers of all atoms, shape (n_atoms,)
        center_type (int): Atomic number of center atom
        cutoff (float): Cutoff distance
    
    Returns:
        np.ndarray: Descriptor vector, shape (32,)
    """
    # Compute distances
    vectors = all_coords - center_coords
    distances = np.linalg.norm(vectors, axis=1)
    
    # Filter neighbors within cutoff (excluding self)
    neighbor_mask = (distances > 1e-6) & (distances < cutoff)
    
    if not neighbor_mask.any():
        return np.zeros(32)
    
    neighbor_vectors = vectors[neighbor_mask]
    neighbor_distances = distances[neighbor_mask]
    neighbor_types = all_types[neighbor_mask]
    
    # Get atomic properties with fallback values
    def get_atomic_property(atomic_num, prop_dict, default=1.0):
        return prop_dict.get(int(atomic_num), default)
    
    center_charge = get_atomic_property(center_type, GAFConfig.ATOMIC_CHARGES, 1.0)
    center_mass = get_atomic_property(center_type, GAFConfig.ATOMIC_MASSES, 12.0)
    center_radius = get_atomic_property(center_type, GAFConfig.ATOMIC_RADII, 0.7)
    
    # Compute RBF features
    rbf = gaussian_rbf(neighbor_distances, n_max=GAFConfig.N_MAX, cutoff=cutoff)
    
    # Compute spherical harmonics
    ylm = compute_spherical_harmonics(neighbor_vectors, l_max=GAFConfig.L_MAX)
    
    # Compute cutoff weights
    cutoff_weights = smooth_cutoff(neighbor_distances, cutoff)
    
    # Weighted features by neighbor type
    descriptor_parts = []
    
    for neighbor_idx in range(len(neighbor_types)):
        neighbor_type = neighbor_types[neighbor_idx]
        neighbor_charge = get_atomic_property(neighbor_type, GAFConfig.ATOMIC_CHARGES, 1.0)
        neighbor_mass = get_atomic_property(neighbor_type, GAFConfig.ATOMIC_MASSES, 12.0)
        neighbor_radius = get_atomic_property(neighbor_type, GAFConfig.ATOMIC_RADII, 0.7)
        
        weight = cutoff_weights[neighbor_idx]
        
        # Feature components
        rbf_weighted = rbf[neighbor_idx] * weight * neighbor_charge
        ylm_weighted = ylm[neighbor_idx] * weight * neighbor_mass
        
        descriptor_parts.append(rbf_weighted)
    
    if len(descriptor_parts) == 0:
        return np.zeros(32)
    
    # Aggregate features
    descriptor = np.concatenate(descriptor_parts)
    
    # Pad or truncate to fixed size
    if len(descriptor) < 32:
        descriptor = np.pad(descriptor, (0, 32 - len(descriptor)), 'constant')
    else:
        descriptor = descriptor[:32]
    
    # Normalize
    norm = np.linalg.norm(descriptor)
    if norm > 1e-10:
        descriptor = descriptor / norm
    
    return descriptor.astype(np.float32)


# ============================================================================
# Coordinate Transformations
# ============================================================================

def cartesian_to_spherical(vectors):
    """
    Convert Cartesian coordinates to spherical coordinates
    
    Args:
        vectors (np.ndarray): Cartesian vectors, shape (n, 3)
    
    Returns:
        tuple: (r, theta, phi) each of shape (n,)
    """
    x, y, z = vectors[:, 0], vectors[:, 1], vectors[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(np.clip(z / (r + 1e-10), -1, 1))
    phi = np.arctan2(y, x)
    return r, theta, phi


def spherical_to_cartesian(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian coordinates
    
    Args:
        r (np.ndarray): Radial distance, shape (n,)
        theta (np.ndarray): Polar angle, shape (n,)
        phi (np.ndarray): Azimuthal angle, shape (n,)
    
    Returns:
        np.ndarray: Cartesian vectors, shape (n, 3)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


# ============================================================================
# Distance and Angle Calculations
# ============================================================================

def compute_distance_matrix(coords):
    """
    Compute pairwise distance matrix
    
    Args:
        coords (np.ndarray): Atomic coordinates, shape (n_atoms, 3)
    
    Returns:
        np.ndarray: Distance matrix, shape (n_atoms, n_atoms)
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    distances = np.sqrt(np.sum(diff**2, axis=2))
    return distances


def compute_angles(vectors):
    """
    Compute angles between consecutive vectors
    
    Args:
        vectors (np.ndarray): Vectors, shape (n, 3)
    
    Returns:
        np.ndarray: Angles in radians, shape (n-1,)
    """
    if len(vectors) < 2:
        return np.array([])
    
    dot_products = np.sum(vectors[:-1] * vectors[1:], axis=1)
    norms1 = np.linalg.norm(vectors[:-1], axis=1)
    norms2 = np.linalg.norm(vectors[1:], axis=1)
    
    cos_angles = dot_products / (norms1 * norms2 + 1e-10)
    angles = np.arccos(np.clip(cos_angles, -1, 1))
    
    return angles


def compute_dihedral_angles(coords):
    """
    Compute dihedral angles for a sequence of 4 atoms
    
    Args:
        coords (np.ndarray): Coordinates of 4 atoms, shape (4, 3)
    
    Returns:
        float: Dihedral angle in radians
    """
    if len(coords) != 4:
        raise ValueError("Need exactly 4 atoms for dihedral angle")
    
    b1 = coords[1] - coords[0]
    b2 = coords[2] - coords[1]
    b3 = coords[3] - coords[2]
    
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    
    n1_norm = n1 / (np.linalg.norm(n1) + 1e-10)
    n2_norm = n2 / (np.linalg.norm(n2) + 1e-10)
    
    cos_angle = np.dot(n1_norm, n2_norm)
    dihedral = np.arccos(np.clip(cos_angle, -1, 1))
    
    return dihedral


# ============================================================================
# Node Feature Builder for GNN
# ============================================================================

class NodeFeatureBuilder:
    """
    Build node features for GNN
    """
    def __init__(self, l_list_dim=10, atomic_emb_dim=16):
        """
        Initialize feature builder
        
        Args:
            l_list_dim (int): Dimension of angular momentum list
            atomic_emb_dim (int): Dimension of atomic embedding
        """
        self.l_list_dim = l_list_dim
        self.atomic_emb_dim = atomic_emb_dim
        self.atomic_embedding = None
    
    def build_features(self, atomic_numbers, max_nu, max_l, l_list, device='cpu'):
        """
        Build node features
        
        Args:
            atomic_numbers (torch.Tensor): Atomic numbers, shape (n_atoms,)
            max_nu (list): Maximum radial quantum numbers
            max_l (list): Maximum angular momentum quantum numbers
            l_list (list): Angular momentum lists
            device (str): Device
        
        Returns:
            torch.Tensor: Node features, shape (n_atoms, feature_dim)
        """
        import torch.nn as nn
        
        if self.atomic_embedding is None:
            self.atomic_embedding = nn.Embedding(100, self.atomic_emb_dim).to(device)
        
        atomic_emb = self.atomic_embedding(atomic_numbers)
        
        max_nu_tensor = torch.tensor(max_nu, dtype=torch.float32, device=device).unsqueeze(1)
        max_l_tensor = torch.tensor(max_l, dtype=torch.float32, device=device).unsqueeze(1)
        l_list_tensor = torch.tensor(l_list, dtype=torch.float32, device=device)
        
        node_feat = torch.cat([atomic_emb, max_nu_tensor, max_l_tensor, l_list_tensor], dim=1)
        
        return node_feat