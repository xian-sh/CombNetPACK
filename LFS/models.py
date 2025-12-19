"""
Neural Network Models
=====================
Contains all model architectures:
- SimpleGNN: Basic graph neural network
- RBFExpansion: Radial basis function expansion
- SchNetInteraction: SchNet interaction layer
- SchNetEncoder: SchNet encoder for molecular representation
- SchNetWithAttention: Final model with attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import build_edge_attr


class SimpleGNN(nn.Module):
    """
    Simple Graph Neural Network for message passing.
    """
    def __init__(self, node_dim, edge_dim=2, hidden_dim=64, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.convs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        
    def forward(self, node_feat, pos, edge_index):
        """
        Forward pass through GNN layers.
        
        Args:
            node_feat (Tensor): Node features, shape (N, node_dim)
            pos (Tensor): Atomic coordinates, shape (N, 3)
            edge_index (Tensor): Edge indices, shape (2, E) - [source, target]
        
        Returns:
            Tensor: Updated node features, shape (N, hidden_dim)
        """
        h = self.node_proj(node_feat)
        row, col = edge_index[0], edge_index[1]
        
        edge_attr = build_edge_attr(pos, row, col)
        edge_emb = self.edge_mlp(edge_attr)
        
        for layer in self.convs:
            h_src = h[row]
            m = torch.cat([h_src, edge_emb], dim=1)
            m = layer(m)
            
            agg = torch.zeros_like(h)
            counts = torch.bincount(col, minlength=h.size(0)).clamp(min=1).float().unsqueeze(1)
            agg.index_add_(0, col, m)
            agg = agg / counts
            
            h = F.relu(h + agg)
            
        return h


class RBFExpansion(nn.Module):
    """
    Radial Basis Function (RBF) expansion for distance features.
    """
    def __init__(self, start=0.0, stop=5.0, num_rbf=20):
        super().__init__()
        self.num_rbf = num_rbf
        self.centers = nn.Parameter(torch.linspace(start, stop, num_rbf))
        self.width = nn.Parameter(torch.tensor(1.0))

    def forward(self, distances):
        """
        Expand distances using RBF.
        
        Args:
            distances (Tensor): Distances, shape (N,)
        
        Returns:
            Tensor: RBF features, shape (N, num_rbf)
        """
        distances = distances.unsqueeze(-1)
        centers = self.centers.unsqueeze(0)
        distances_expanded = torch.exp(-((distances - centers) ** 2) / (self.width ** 2))
        return distances_expanded


class SchNetInteraction(nn.Module):
    """
    SchNet interaction layer for message passing.
    """
    def __init__(self, hidden_dim=64, n_rbf=20):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        
        self.output_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, h, edge_index, edge_attr, mask):
        """
        Forward pass through interaction layer.
        
        Args:
            h (Tensor): Atom features, shape (N, hidden_dim)
            edge_index (Tensor): Edge indices, shape (2, E)
            edge_attr (Tensor): Edge features (RBF), shape (E, n_rbf)
            mask (Tensor): Mask (currently unused)
        
        Returns:
            Tensor: Updated atom features, shape (N, hidden_dim)
        """
        row, col = edge_index[0], edge_index[1]
        
        h_row = h[row]
        h_col = h[col]
        
        edge_weight = self.interaction_mlp(edge_attr)
        msg = h_row * edge_weight
        
        h_new = torch.zeros_like(h)
        h_new.index_add_(0, col, msg)
        
        h_combined = torch.cat([h, h_new], dim=1)
        h_out = self.output_mlp(h_combined)
        
        return h + h_out


class SchNetEncoder(nn.Module):
    """
    SchNet encoder for molecular representation learning.
    """
    def __init__(self, input_dim=68, hidden_dim=64, output_dim=128, n_interactions=3, n_rbf=20):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.rbf_expansion = RBFExpansion(start=0.0, stop=5.0, num_rbf=n_rbf)
        
        self.interactions = nn.ModuleList([
            SchNetInteraction(hidden_dim=hidden_dim, n_rbf=n_rbf)
            for _ in range(n_interactions)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, atomic_features, mask):
        """
        Forward pass through SchNet encoder.
        
        Args:
            atomic_features (Tensor): Atomic features, shape (B, max_len, 68)
            mask (Tensor): Mask for valid atoms, shape (B, max_len)
        
        Returns:
            Tensor: Molecular representation, shape (B, output_dim)
        """
        B, max_len, _ = atomic_features.shape
        
        h = atomic_features.view(-1, self.input_dim)
        h = self.input_proj(h)
        
        coords = atomic_features[:, :, 1:4]
        coords_flat = coords.view(-1, 3)
        
        all_h_out = []
        
        for b_idx in range(B):
            current_mask = mask[b_idx]
            n_atoms = current_mask.sum().item()
            
            if n_atoms <= 1:
                h_batch = h[b_idx * max_len:(b_idx + 1) * max_len]
                all_h_out.append(h_batch)
                continue
            
            h_current = h[b_idx * max_len:(b_idx * max_len + n_atoms)]
            coords_current = coords_flat[b_idx * max_len:(b_idx * max_len + n_atoms)]
            
            dist_matrix = torch.cdist(coords_current, coords_current)
            
            cutoff = 5.0
            edge_mask = (dist_matrix > 0) & (dist_matrix < cutoff)
            edge_idx = torch.nonzero(edge_mask, as_tuple=False).t()
            
            if edge_idx.size(1) == 0:
                if n_atoms >= 2:
                    edge_idx = torch.tensor([[0, 1], [1, 0]], dtype=torch.long, device=h.device)
                else:
                    edge_idx = torch.tensor([[0, 0], [0, 0]], dtype=torch.long, device=h.device)
            
            row, col = edge_idx[0], edge_idx[1]
            edge_dist = dist_matrix[row, col]
            edge_attr = self.rbf_expansion(edge_dist)
            
            h_current = h_current
            for interaction in self.interactions:
                h_current = interaction(h_current, edge_idx, edge_attr, None)
            
            h_batch = torch.zeros(max_len, self.hidden_dim, device=h.device, dtype=h.dtype)
            h_batch[:n_atoms] = h_current
            all_h_out.append(h_batch)
        
        h = torch.stack(all_h_out, dim=0)
        h = h * mask.unsqueeze(-1).float()
        h = self.output_proj(h)
        mol_repr = torch.sum(h, dim=1)
        
        return mol_repr


class SchNetWithAttention(nn.Module):
    """
    SchNet-based model with attention mechanism for LFS prediction.
    """
    def __init__(self, config):
        super().__init__()
        self.D_mol = config.D_MOL
        self.D_cond = config.D_COND
        self.D_attn = config.D_ATTN
        
        self.schnet_encoder = SchNetEncoder(
            input_dim=config.ATOMIC_FEATURE_DIM,
            hidden_dim=config.SCHNET_HIDDEN_DIM,
            output_dim=self.D_mol,
            n_interactions=config.N_INTERACTIONS,
            n_rbf=config.N_RBF
        )
        
        self.proj_q = nn.Linear(self.D_mol, self.D_attn)
        self.proj_k = nn.Linear(self.D_cond, self.D_attn)
        self.proj_v = nn.Linear(self.D_cond, self.D_attn)
        
        self.final_mlp = nn.Sequential(
            nn.Linear(self.D_mol + self.D_attn, config.FINAL_MLP_HIDDEN1),
            nn.SiLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.FINAL_MLP_HIDDEN1, config.FINAL_MLP_HIDDEN2),
            nn.SiLU(),
            nn.Linear(config.FINAL_MLP_HIDDEN2, 1)
        )

    def forward(self, atomic_features, mask, conditions):
        """
        Forward pass.
        
        Args:
            atomic_features (Tensor): Atomic features, shape (B, max_len, 68)
            mask (Tensor): Mask for valid atoms, shape (B, max_len)
            conditions (Tensor): Operating conditions, shape (B, 3)
        
        Returns:
            Tensor: LFS predictions, shape (B,)
        """
        v_mol = self.schnet_encoder(atomic_features, mask)
        
        Q = self.proj_q(v_mol).unsqueeze(1)
        K = self.proj_k(conditions).unsqueeze(1)
        V = self.proj_v(conditions).unsqueeze(1)

        attn_scores = torch.bmm(Q, K.transpose(1, 2)) / (self.D_attn ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=2)
        v_att = torch.bmm(attn_weights, V).squeeze(1)
        
        fused_vec = torch.cat([v_mol, v_att], dim=-1)
        prediction = self.final_mlp(fused_vec).squeeze(-1)
        
        return prediction