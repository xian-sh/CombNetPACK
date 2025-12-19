"""
Data Processing Module
======================
Functions and classes for:
- Building atomic features from molecules
- Processing NPZ files
- Creating PyTorch datasets and dataloaders
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from config import Config
from features import NodeFeatureBuilder, compute_descriptor
from models import SimpleGNN


def build_atomic_features_for_molecule(pos_dict, T_norm, p_norm, phi_norm, lfs_value, logger=None):
    """
    Build complete atomic features for a single molecule.
    
    Features include:
    - Atomic type (1D)
    - 3D coordinates (3D)
    - GNN features (32D)
    - Atom C descriptor (32D)
    
    Args:
        pos_dict (dict): Dictionary with 'elements' and 'positions'
        T_norm (float): Normalized temperature
        p_norm (float): Normalized pressure
        phi_norm (float): Normalized equivalence ratio
        lfs_value (float): Target LFS value
        logger (logging.Logger): Logger instance
    
    Returns:
        tuple: (final_features, conditions, target)
    """
    elements = pos_dict["elements"]
    positions = pos_dict["positions"]
    N = len(elements)
    
    element_to_idx = {
        'H': 0, 'C': 1, 'N': 2, 'O': 3, 'S': 4, 
        'F': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'P': 9
    }
    atomic_numbers = [element_to_idx.get(e, 0) for e in elements]
    
    pos_tensor = torch.tensor(positions, dtype=torch.float32)
    dist_matrix = torch.cdist(pos_tensor, pos_tensor)
    
    cutoff = Config.CUTOFF
    edge_mask = (dist_matrix > 0) & (dist_matrix < cutoff)
    edge_index = torch.nonzero(edge_mask, as_tuple=False).t()
    
    if edge_index.size(1) == 0:
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    max_nu = [1.0] * N
    max_l = [2.0] * N
    l_list = np.random.rand(N, Config.L_LIST_DIM).astype(np.float32)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    node_builder = NodeFeatureBuilder(l_list_dim=Config.L_LIST_DIM, 
                                     atomic_emb_dim=Config.ATOMIC_EMB_DIM)
    gnn = SimpleGNN(node_dim=Config.ATOMIC_EMB_DIM + 2 + Config.L_LIST_DIM, 
                   hidden_dim=Config.GNN_HIDDEN_DIM, 
                   num_layers=Config.GNN_NUM_LAYERS)
    
    node_builder = node_builder.to(device)
    gnn = gnn.to(device)
    
    node_feat = node_builder(
        atomic_numbers=atomic_numbers,
        max_nu=max_nu,
        max_l=max_l,
        l_list=l_list,
        device=device
    )
    
    with torch.no_grad():
        gnn_features = gnn(
            node_feat=node_feat,
            pos=pos_tensor.to(device),
            edge_index=edge_index.to(device)
        ).cpu().detach().numpy()
    
    atom_c_features = []
    for i, (center_coords, center_type) in enumerate(zip(positions, elements)):
        desc = compute_descriptor(
            center_coords=center_coords,
            all_coords=positions,
            all_types=elements,
            center_type=center_type,
            cutoff=Config.ATOM_C_CUTOFF
        )
        atom_c_features.append(desc)
    atom_c_features = np.array(atom_c_features)
    
    atomic_nums_array = np.array(atomic_numbers).reshape(-1, 1)
    final_features = np.concatenate([
        atomic_nums_array,
        positions,
        gnn_features,
        atom_c_features
    ], axis=1)
    
    conditions = np.array([T_norm, p_norm, phi_norm])
    
    return final_features, conditions, lfs_value


def process_npz_file_for_features(input_npz_path, output_npz_path, use_compression=True, logger=None):
    """
    Process NPZ file to compute atomic features and save.
    
    Args:
        input_npz_path (str): Path to input NPZ file with 3D coordinates
        output_npz_path (str): Path to save processed features
        use_compression (bool): Whether to use compression when saving
        logger (logging.Logger): Logger instance
    """
    if logger:
        logger.info("=" * 60)
        logger.info("Starting NPZ file processing for atomic features...")
        logger.info("=" * 60)
    
    data = np.load(input_npz_path, allow_pickle=True)
    smiles_list = data["smiles"]
    pos_data_list = data["pos"]
    T_norm_list = data["T_norm"]
    p_norm_list = data["p_norm"]
    phi_norm_list = data["phi_norm"]
    lfs_list = data["LFS (cm/s)"]
    
    if logger:
        logger.info(f"Loaded {len(smiles_list)} molecules")
    
    all_molecular_features = []
    all_conditions = []
    all_targets = []
    
    for i in tqdm(range(len(smiles_list)), desc="Processing molecules"):
        pos_dict = pos_data_list[i]
        T_norm = T_norm_list[i]
        p_norm = p_norm_list[i]
        phi_norm = phi_norm_list[i]
        lfs_value = lfs_list[i]
        
        try:
            mol_features, conditions, target = build_atomic_features_for_molecule(
                pos_dict, T_norm, p_norm, phi_norm, lfs_value, logger
            )
            all_molecular_features.append(mol_features)
            all_conditions.append(conditions)
            all_targets.append(target)
        except Exception as e:
            if logger:
                logger.error(f"Error processing molecule {i} ({smiles_list[i]}): {e}")
            continue
    
    if logger:
        logger.info(f"Successfully processed {len(all_molecular_features)} molecules")
    
    if logger:
        logger.info("Normalizing coordinates...")
    
    all_coords = []
    for mol_feat in all_molecular_features:
        coords = mol_feat[:, 1:4]
        all_coords.append(coords)
    
    all_coords_combined = np.vstack(all_coords)
    
    coord_scaler = MinMaxScaler()
    all_coords_normalized = coord_scaler.fit_transform(all_coords_combined)
    
    start_idx = 0
    for j, mol_feat in enumerate(all_molecular_features):
        n_atoms = mol_feat.shape[0]
        mol_feat[:, 1:4] = all_coords_normalized[start_idx:start_idx+n_atoms]
        start_idx += n_atoms
    
    processed_data = {
        "molecular_features": np.array(all_molecular_features, dtype=object),
        "conditions": np.array(all_conditions),
        "targets": np.array(all_targets),
        "smiles": np.array(smiles_list[:len(all_molecular_features)]),
        "coord_scaler_params": {
            "min_": coord_scaler.min_,
            "scale_": coord_scaler.scale_,
            "data_min_": coord_scaler.data_min_,
            "data_max_": coord_scaler.data_max_
        }
    }
    
    if use_compression:
        np.savez_compressed(output_npz_path, **processed_data)
    else:
        np.savez(output_npz_path, **processed_data)
        
    size_mb = os.path.getsize(output_npz_path) / (1024 * 1024)
    if logger:
        logger.info(f"Saved processed data to: {output_npz_path} ({size_mb:.2f} MB)")
    
    loaded_data = np.load(output_npz_path, allow_pickle=True)
    if logger:
        logger.info(f"Keys in NPZ file: {list(loaded_data.keys())}")
        logger.info(f"Number of molecules: {len(loaded_data['molecular_features'])}")
        if len(loaded_data['molecular_features']) > 0:
            logger.info(f"Number of atoms in first molecule: {loaded_data['molecular_features'][0].shape[0]}")
            logger.info(f"Feature dimension of first molecule: {loaded_data['molecular_features'][0].shape[1]}")
            logger.info(f"Feature sample (first atom): {loaded_data['molecular_features'][0][0][:5]}...")
            logger.info(f"Operating conditions: {loaded_data['conditions'][0]}")
            logger.info(f"LFS value: {loaded_data['targets'][0]}")


class LFS_FeatureDataset(Dataset):
    """
    Dataset for LFS prediction with atomic features.
    """
    def __init__(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        self.molecular_features = data["molecular_features"]
        self.conditions = data["conditions"]
        self.targets = data["targets"]
        self.smiles = data["smiles"]
        
        self.size = len(self.molecular_features)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        mol_features = self.molecular_features[idx].astype(np.float32)
        conditions = self.conditions[idx].astype(np.float32)
        target = np.array(self.targets[idx], dtype=np.float32)
        
        return mol_features, conditions, target


def collate_fn_new(batch):
    """
    Collate function to handle variable-length molecules with padding and masking.
    
    Args:
        batch (list): List of (mol_features, conditions, target) tuples
    
    Returns:
        tuple: (padded_mol_features, masks, conditions, targets)
    """
    mol_features_list, conditions_list, target_list = zip(*batch)
    
    max_len = Config.MAX_ATOMS
    
    padded_mol_features = []
    masks = []
    
    for features in mol_features_list:
        n_atoms = features.shape[0]
        
        if n_atoms > max_len:
            features_padded = features[:max_len, :]
            mask = np.ones(max_len, dtype=np.bool_)
        else:
            pad_len = max_len - n_atoms
            features_padded = np.pad(features, ((0, pad_len), (0, 0)), 
                                   'constant', constant_values=0.0)
            mask = np.array([True] * n_atoms + [False] * pad_len, dtype=np.bool_)

        padded_mol_features.append(features_padded)
        masks.append(mask)

    return (
        torch.tensor(np.stack(padded_mol_features), dtype=torch.float32),
        torch.tensor(np.stack(masks), dtype=torch.bool),
        torch.tensor(np.stack(conditions_list), dtype=torch.float32),
        torch.tensor(np.stack(target_list), dtype=torch.float32)
    )