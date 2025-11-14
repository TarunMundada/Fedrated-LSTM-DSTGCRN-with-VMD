"""
LSTM-DSTGCRN (Long Short-Term Memory - Dynamic Spatial-Temporal Graph Convolutional Recurrent Network) model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from .graph_components import AdaptiveAdjacency
from .lstm_cells import GraphConvLSTMCell, HybridGCRNGATCell

class LSTMDSTGCRN(nn.Module):
    """
    x: [B,T_in,N,C_in] -> y: [B,T_out,N,C_out]
    If using VMD and predicting IMFs: C_out = K (number of modes). Otherwise C_out = 1.
    
    Supports three spatial modeling modes:
    1. GCRN-only (default): use_gat=False
    2. GAT-only: use_gat=True, use_gcrn=False
    3. Hybrid: use_gat=True, use_gcrn=True (with fusion)
    """
    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        horizon: int = 3,
        num_layers: int = 1,
        emb_dim: int = 16,
        use_attention: bool = False,
        num_heads: int = 2,
        ablate_adja: bool = False,
        # GAT-specific parameters
        use_gat: bool = False,
        use_gcrn: bool = True,
        gat_heads: int = 4,
        gat_dropout: float = 0.1,
        gat_fusion_mode: str = 'concat',
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.horizon = horizon
        self.use_gat = use_gat
        self.use_gcrn = use_gcrn
        
        # Ensure at least one spatial modeling approach is enabled
        if not use_gat and not use_gcrn:
            # Default to GCRN if neither is specified
            self.use_gcrn = True
            print("Warning: Neither GAT nor GCRN enabled. Defaulting to GCRN.")
        
        self.adaptiveA = AdaptiveAdjacency(num_nodes, emb_dim, ablate=ablate_adja)
        self.use_attention = use_attention
        if use_attention:
            assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
            # Using hidden_dim for attention, as input is projected to hidden_dim first
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        else:
            self.attn = None

        # Build LSTM cells with appropriate spatial modeling
        cells = []
        for l in range(num_layers):
            in_dim = hidden_dim if self.use_attention else (input_dim if l == 0 else hidden_dim)
            
            if self.use_gat:
                # Use hybrid cell that supports GAT, GCRN, or both
                cell = HybridGCRNGATCell(
                    num_nodes=num_nodes,
                    in_dim=in_dim, 
                    hidden_dim=hidden_dim,
                    gat_heads=gat_heads,
                    gat_dropout=gat_dropout,
                    fusion_mode=gat_fusion_mode,
                    use_gcrn=self.use_gcrn,
                    use_gat=self.use_gat
                )
            else:
                # Use traditional GCRN-only cell
                cell = GraphConvLSTMCell(num_nodes, in_dim, hidden_dim)
            
            cells.append(cell)
        
        self.cells = nn.ModuleList(cells)
        self.proj = nn.Linear(hidden_dim, output_dim * horizon)

        # expose groups for selective integration
        self.modules_to_integrate = {
            "adaptiveA": self.adaptiveA,
            "cells": self.cells,
            "proj": self.proj,
        }
        if self.attn is not None:
            self.modules_to_integrate["attn"] = self.attn
            self.modules_to_integrate["input_proj"] = self.input_proj

    def forward(self, x):
        # x: [B,T,N,C]
        B, T, N, C = x.shape
        assert N == self.num_nodes, f"Input N={N} but model expects {self.num_nodes}"
        A = self.adaptiveA()  # [N,N]

        if self.attn is not None:
            x = self.input_proj(x) # Project to hidden_dim
            x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, -1)
            x_attn, _ = self.attn(x_flat, x_flat, x_flat)
            x = x_attn.reshape(B, N, T, -1).permute(0, 2, 1, 3)

        hs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]
        cs = [x.new_zeros(B, N, c.hidden_dim) for c in self.cells]

        for t in range(T):
            xt = x[:, t]
            for i, cell in enumerate(self.cells):
                h, c = cell(xt, hs[i], cs[i], A)
                hs[i], cs[i] = h, c
                xt = h

        out = self.proj(hs[-1])  # [B,N, C_out*horizon]
        out = out.view(B, N, self.horizon, -1).permute(0, 2, 1, 3)  # [B,T_out,N,C_out]
        return out