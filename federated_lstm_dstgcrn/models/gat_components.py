"""
Graph Attention Network (GAT) components.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class GraphAttentionLayer(nn.Module):
    """Single-head Graph Attention Layer"""
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.1, 
                 alpha: float = 0.2, concat: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat
        
        # Linear transformation
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, h: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        h: [B, N, in_features]
        adj: [N, N] adjacency matrix (optional, used for masking)
        Returns: [B, N, out_features]
        """
        B, N, _ = h.shape
        
        # Linear transformation: [B, N, out_features]
        Wh = torch.matmul(h, self.W)
        
        # Compute attention coefficients
        # Create all pairs: [B, N, N, 2*out_features]
        Wh1 = Wh.unsqueeze(2).expand(-1, -1, N, -1)  # [B, N, N, out_features]
        Wh2 = Wh.unsqueeze(1).expand(-1, N, -1, -1)  # [B, N, N, out_features]
        concat_features = torch.cat([Wh1, Wh2], dim=3)  # [B, N, N, 2*out_features]
        
        # Apply attention mechanism: [B, N, N]
        e = self.leakyrelu(torch.matmul(concat_features, self.a).squeeze(3))
        
        # Mask with adjacency matrix if provided
        if adj is not None:
            # Convert adjacency to mask (0 where no edge, -inf where edge doesn't exist)
            mask = (adj == 0).float() * -1e9
            e = e + mask.unsqueeze(0)  # Broadcast to batch dimension
        
        # Compute attention weights
        attention = F.softmax(e, dim=2)  # [B, N, N]
        attention = self.dropout_layer(attention)
        
        # Apply attention to features
        h_prime = torch.matmul(attention, Wh)  # [B, N, out_features]
        
        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

class MultiHeadGAT(nn.Module):
    """Multi-head Graph Attention Network"""
    def __init__(self, in_features: int, out_features: int, num_heads: int = 4, 
                 dropout: float = 0.1, alpha: float = 0.2):
        super().__init__()
        self.num_heads = num_heads
        self.out_features = out_features
        
        # Ensure out_features is divisible by num_heads
        assert out_features % num_heads == 0, f"out_features ({out_features}) must be divisible by num_heads ({num_heads})"
        self.head_dim = out_features // num_heads
        
        # Create attention heads
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(in_features, self.head_dim, dropout=dropout, alpha=alpha, concat=True)
            for _ in range(num_heads)
        ])
        
        # Output projection
        self.out_proj = nn.Linear(out_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, adj: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, N, in_features]
        adj: [N, N] adjacency matrix (optional)
        Returns: [B, N, out_features]
        """
        # Apply each attention head
        head_outputs = []
        for attention in self.attentions:
            head_out = attention(x, adj)  # [B, N, head_dim]
            head_outputs.append(head_out)
        
        # Concatenate heads: [B, N, out_features]
        x = torch.cat(head_outputs, dim=2)
        
        # Apply output projection and dropout
        x = self.out_proj(x)
        x = self.dropout(x)
        
        return x