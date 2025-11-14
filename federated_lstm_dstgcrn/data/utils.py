"""
Data utilities for handling batches and padding.
"""
import torch
from typing import List, Tuple

def collate_batch(batch):
    """Collate function for DataLoader."""
    xs, y_imfs, y_trips = zip(*batch)
    return torch.stack(xs, 0), (torch.stack(y_imfs, 0), torch.stack(y_trips, 0))

def pad_batch_to_full(x_sub: torch.Tensor, subset_idx: List[int], N_full: int):
    """
    x_sub: [B,T,N_sub,C]
    returns x_full: [B,T,N_full,C], mask: [B,T,N_full,1]
    """
    B, T, N_sub, C = x_sub.shape
    x_full = x_sub.new_zeros((B, T, N_full, C))
    mask = x_sub.new_zeros((B, T, N_full, 1))
    for i, pos in enumerate(subset_idx):
        x_full[:, :, pos, :] = x_sub[:, :, i, :]
        mask[:, :, pos, 0] = 1.0
    return x_full, mask