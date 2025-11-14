"""
LSTM cell components with graph convolutions.
"""
import torch
import torch.nn as nn
from typing import Tuple
from .graph_components import GraphConv
from .gat_components import MultiHeadGAT

class GraphConvLSTMCell(nn.Module):
    """Graph Convolutional LSTM Cell"""
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gc_x = GraphConv(in_dim, 4 * hidden_dim)
        self.gc_h = GraphConv(hidden_dim, 4 * hidden_dim)

    def forward(self, x_t, h_prev, c_prev, A):
        gates = self.gc_x(x_t, A) + self.gc_h(h_prev, A)  # [B,N,4H]
        i, f, o, g = torch.chunk(gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        return h_t, c_t

class HybridGCRNGATCell(nn.Module):
    """Hybrid GCRN + GAT LSTM Cell implementing parallel spatial branches with fusion"""
    def __init__(self, num_nodes: int, in_dim: int, hidden_dim: int, 
                 gat_heads: int = 4, gat_dropout: float = 0.1, 
                 fusion_mode: str = 'concat', use_gcrn: bool = True, use_gat: bool = True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.fusion_mode = fusion_mode
        self.use_gcrn = use_gcrn
        self.use_gat = use_gat
        
        # Ensure at least one spatial branch is enabled
        assert use_gcrn or use_gat, "At least one of GCRN or GAT must be enabled"
        
        # GCRN branch components
        if self.use_gcrn:
            self.gc_x_gcrn = GraphConv(in_dim, 4 * hidden_dim)
            self.gc_h_gcrn = GraphConv(hidden_dim, 4 * hidden_dim)
        
        # GAT branch components
        if self.use_gat:
            self.gat_x = MultiHeadGAT(in_dim, 4 * hidden_dim, num_heads=gat_heads, dropout=gat_dropout)
            self.gat_h = MultiHeadGAT(hidden_dim, 4 * hidden_dim, num_heads=gat_heads, dropout=gat_dropout)
        
        # Fusion layer
        if self.use_gcrn and self.use_gat:
            if fusion_mode == 'concat':
                # Concatenate outputs and project back to hidden_dim
                self.fusion_proj = nn.Linear(8 * hidden_dim, 4 * hidden_dim)
            elif fusion_mode == 'add':
                # Element-wise addition (requires same dimensions)
                pass  # No additional parameters needed
            elif fusion_mode == 'weighted':
                # Learnable weighted combination
                self.gcrn_weight = nn.Parameter(torch.tensor(0.5))
                self.gat_weight = nn.Parameter(torch.tensor(0.5))
            else:
                raise ValueError(f"Unknown fusion_mode: {fusion_mode}. Choose from ['concat', 'add', 'weighted']")
    
    def forward(self, x_t, h_prev, c_prev, A):
        """
        Apply parallel GCRN and GAT spatial processing followed by fusion.
        
        Args:
            x_t: [B, N, F] - input features at time t
            h_prev: [B, N, H] - previous hidden state
            c_prev: [B, N, H] - previous cell state
            A: [N, N] - adjacency matrix
        
        Returns:
            h_t: [B, N, H] - new hidden state
            c_t: [B, N, H] - new cell state
        """
        
        spatial_outputs = []
        
        # GCRN branch: traditional graph convolution
        if self.use_gcrn:
            gates_gcrn = self.gc_x_gcrn(x_t, A) + self.gc_h_gcrn(h_prev, A)  # [B,N,4H]
            spatial_outputs.append(gates_gcrn)
        
        # GAT branch: attention-based spatial modeling
        if self.use_gat:
            # Convert adjacency matrix to edge connectivity for attention masking
            # Use A > 0 to create a binary adjacency for masking
            adj_mask = (A > 1e-6).float() if A is not None else None
            
            gates_gat_x = self.gat_x(x_t, adj_mask)    # [B,N,4H]
            gates_gat_h = self.gat_h(h_prev, adj_mask) # [B,N,4H]
            gates_gat = gates_gat_x + gates_gat_h      # [B,N,4H]
            spatial_outputs.append(gates_gat)
        
        # Fusion step
        if len(spatial_outputs) == 0:
            # This should not happen given our assertions, but handle gracefully
            raise RuntimeError("No spatial outputs generated. Check use_gcrn and use_gat flags.")
        elif len(spatial_outputs) == 1:
            # Single branch (either GCRN-only or GAT-only)
            fused_gates = spatial_outputs[0]
        else:
            # Multiple branches - apply fusion
            gates_gcrn, gates_gat = spatial_outputs[0], spatial_outputs[1]
            
            if self.fusion_mode == 'concat':
                # Concatenate and project back
                concat_gates = torch.cat([gates_gcrn, gates_gat], dim=-1)  # [B,N,8H]
                fused_gates = self.fusion_proj(concat_gates)  # [B,N,4H]
            elif self.fusion_mode == 'add':
                # Element-wise addition
                fused_gates = gates_gcrn + gates_gat  # [B,N,4H]
            elif self.fusion_mode == 'weighted':
                # Learnable weighted combination
                # Normalize weights to sum to 1
                total_weight = self.gcrn_weight + self.gat_weight
                w_gcrn = self.gcrn_weight / total_weight
                w_gat = self.gat_weight / total_weight
                fused_gates = w_gcrn * gates_gcrn + w_gat * gates_gat  # [B,N,4H]
            else:
                raise ValueError(f"Unknown fusion_mode: {self.fusion_mode}")
        
        # Apply LSTM gates
        i, f, o, g = torch.chunk(fused_gates, 4, dim=-1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        
        # Update cell and hidden states
        c_t = f * c_prev + i * g
        h_t = o * torch.tanh(c_t)
        
        return h_t, c_t