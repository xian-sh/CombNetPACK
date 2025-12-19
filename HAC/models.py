"""
Neural Network Models
=====================
GNN and CombNet models for GAF features and velocity prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


# ============================================================================
# GNN Models for GAF Feature Extraction
# ============================================================================

class SimpleGNN(nn.Module):
    """
    Simple Graph Neural Network for molecular feature learning
    Used in GAF feature extraction
    """
    def __init__(self, node_dim, hidden_dim=32, num_layers=3):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GNN layers
        self.conv_layers = nn.ModuleList([
            SimpleConvLayer(hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 32)
    
    def forward(self, node_feat, pos, edge_index):
        """
        Forward pass
        
        Args:
            node_feat (Tensor): Node features, shape (n_nodes, node_dim)
            pos (Tensor): Node positions, shape (n_nodes, 3)
            edge_index (Tensor): Edge indices, shape (2, n_edges)
        
        Returns:
            Tensor: Output features, shape (n_nodes, 32)
        """
        x = self.node_encoder(node_feat)
        
        for conv in self.conv_layers:
            x = conv(x, edge_index, pos)
        
        out = self.output_proj(x)
        return out


class SimpleConvLayer(MessagePassing):
    """
    Simple convolution layer for GNN
    """
    def __init__(self, hidden_dim):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
    
    def forward(self, x, edge_index, pos):
        """
        Forward pass
        
        Args:
            x (Tensor): Node features, shape (n_nodes, hidden_dim)
            edge_index (Tensor): Edge indices, shape (2, n_edges)
            pos (Tensor): Node positions, shape (n_nodes, 3)
        
        Returns:
            Tensor: Updated node features, shape (n_nodes, hidden_dim)
        """
        out = self.propagate(edge_index, x=x, pos=pos)
        out = self.update_mlp(torch.cat([x, out], dim=-1))
        return out
    
    def message(self, x_i, x_j, pos_i, pos_j):
        """
        Compute messages
        
        Args:
            x_i (Tensor): Target node features
            x_j (Tensor): Source node features
            pos_i (Tensor): Target node positions
            pos_j (Tensor): Source node positions
        
        Returns:
            Tensor: Messages
        """
        dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
        msg_input = torch.cat([x_i, x_j, dist], dim=-1)
        msg = self.message_mlp(msg_input)
        return msg


# ============================================================================
# CombNet Models for Velocity Prediction
# ============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    def __init__(self, d_model, n_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x (Tensor): Input features, shape (B, seq_len, d_model)
            mask (Tensor): Attention mask, shape (B, seq_len)
        
        Returns:
            Tensor: Attention output, shape (B, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Linear projections and reshape
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        # Apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, seq_len)
            scores = scores.masked_fill(~mask, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # Concatenate heads and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(attn_output)
        
        return output


class CombNetEncoderLayer(nn.Module):
    """
    Single CombNet encoder layer
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Forward pass
        
        Args:
            x (Tensor): Input features, shape (B, seq_len, d_model)
            mask (Tensor): Attention mask, shape (B, seq_len)
        
        Returns:
            Tensor: Output features, shape (B, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x


class AtomicEncoder(nn.Module):
    """
    Atomic feature encoder
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (Tensor): Input features, shape (B, n_atoms, input_dim)
        
        Returns:
            Tensor: Encoded features, shape (B, n_atoms, hidden_dim)
        """
        return self.encoder(x)


class VelocityCombNet(nn.Module):
    """
    CombNet-based model for velocity prediction
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Atomic feature encoder
        self.atomic_encoder = AtomicEncoder(
            config.TOTAL_ATOM_DIM, 
            config.HIDDEN_DIM, 
            config.DROPOUT
        )
        
        # CombNet encoder layers
        self.transformer_layers = nn.ModuleList([
            CombNetEncoderLayer(
                config.HIDDEN_DIM, 
                config.N_HEADS, 
                config.DROPOUT
            )
            for _ in range(config.N_LAYERS)
        ])
        
        # Velocity prediction head
        self.velocity_head = nn.Sequential(
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM // 2, config.VELOCITY_DIM)
        )
        
    def forward(self, features, masks):
        """
        Forward pass
        
        Args:
            features (Tensor): Atom features, shape (B, max_atoms, 64)
            masks (Tensor): Valid atom masks, shape (B, max_atoms)
        
        Returns:
            Tensor: Predicted velocities, shape (B, max_atoms, 3)
        """
        # Encode atomic features
        atom_embeddings = self.atomic_encoder(features)  # (B, max_atoms, hidden_dim)
        
        # Apply transformer layers
        x = atom_embeddings
        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x, masks)
        
        # Predict velocities
        velocities = self.velocity_head(x)  # (B, max_atoms, 3)
        
        return velocities

