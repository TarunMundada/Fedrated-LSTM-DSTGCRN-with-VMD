"""
Selective integration for federated learning.
"""
import copy
import math
from typing import Dict, List, Optional, Tuple
import torch
from ..training.trainer import evaluate

def _copy_group(dst: torch.nn.Module, src: torch.nn.Module, group_name: str):
    """Copy a specific group of parameters from source to destination model."""
    dst_group = getattr(dst, 'modules_to_integrate')[group_name]
    src_group = getattr(src, 'modules_to_integrate')[group_name]
    dst_group.load_state_dict(src_group.state_dict())

def selective_integration(local_model: torch.nn.Module, global_model: torch.nn.Module, val_loader, device, mase_scaler: float, subset_idx: Optional[List[int]] = None, N_full: Optional[int] = None, groups: Optional[List[str]] = None):
    """Selectively integrate global model components that improve local performance."""
    if groups is None:
        groups = list(getattr(local_model, 'modules_to_integrate').keys())

    best_loss = math.inf
    best_group = None
    base = copy.deepcopy(local_model)

    # First, evaluate the local model as a baseline
    base_mae, _, _, _, _, _, _ = evaluate(base, val_loader, device, mase_scaler, subset_idx=subset_idx, N_full=N_full)
    best_loss = base_mae

    for g in groups:
        cand = copy.deepcopy(base)
        _copy_group(cand, global_model, g)
        val_mae, _, _, _, _, _, _ = evaluate(cand, val_loader, device, mase_scaler, subset_idx=subset_idx, N_full=N_full)
        if val_mae < best_loss:
            best_loss, best_group = val_mae, g

    if best_group is not None:
        _copy_group(local_model, global_model, best_group)
    
    return local_model, best_group