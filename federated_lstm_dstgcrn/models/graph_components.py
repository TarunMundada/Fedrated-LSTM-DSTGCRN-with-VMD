"""
Graph components for the LSTM-DSTGCRN model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class AdaptiveAdjacency(nn.Module):
    """Adaptive adjacency matrix module."""
    def __init__(self, num_nodes: int, emb_dim: int, ablate: bool = False):
        super().__init__()
        self.ablate = ablate
        if not self.ablate:
            self.E1 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
            self.E2 = nn.Parameter(torch.randn(num_nodes, emb_dim) * 0.1)
        self.num_nodes = num_nodes

    def forward(self):
        if self.ablate:
            # Return identity matrix for ablation study
            return torch.eye(self.num_nodes, device=self.E1.device if hasattr(self, 'E1') else 'cpu')
        logits = F.relu(self.E1 @ self.E2.t())  # [N,N]
        return F.softmax(logits, dim=-1)

class GraphConv(nn.Module):
    """Graph convolution layer."""
    def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=bias)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        # x: [B,N,C], A: [N,N]
        x_mp = torch.einsum("ij,bjc->bic", A, x)
        return self.lin(x_mp)