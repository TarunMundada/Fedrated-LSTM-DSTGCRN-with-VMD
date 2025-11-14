"""
Training and evaluation functions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
from ..utils.general_utils import to_device
from ..data.utils import pad_batch_to_full
from .metrics import _mae

def train_one_epoch(model, loader, optimizer, device, subset_idx: Optional[List[int]] = None, N_full: Optional[int] = None):
    """Train model for one epoch."""
    model.train()
    total_mae = 0.0
    n_batches = 0
    
    print(f"Starting training epoch with {len(loader)} batches")

    for batch_idx, (x, (y_imf, y_trips)) in enumerate(loader):
        if batch_idx % 5 == 0:
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")
        
        # x: [B,T_in,N,F], y_imf: [B,T_out,N,Kor1], y_trips: [B,T_out,N,1]
        mask = None
        if subset_idx is not None and N_full is not None:
            x, _ = pad_batch_to_full(x, subset_idx, N_full)
            y_imf, _ = pad_batch_to_full(y_imf, subset_idx, N_full)
            y_trips, mask = pad_batch_to_full(y_trips, subset_idx, N_full)
        
        x, y_imf, y_trips = to_device(x, device), to_device(y_imf, device), to_device(y_trips, device)
        if mask is not None: mask = to_device(mask, device)

        optimizer.zero_grad()
        
        try:
            y_hat_imf = model(x)                      # [B,T_out,N,Kor1]
            y_hat_trips = y_hat_imf.sum(dim=-1, keepdim=True)  # reconstruct trips

            loss = _mae(y_hat_trips, y_trips, mask=mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Reduced gradient clipping
            optimizer.step()

            total_mae += loss.item()
            n_batches += 1
            
            # Clear GPU cache periodically
            if torch.cuda.is_available() and batch_idx % 3 == 0:
                torch.cuda.empty_cache()
                
        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    return total_mae / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, loader, device, mase_scaler: float, subset_idx: Optional[List[int]] = None, N_full: Optional[int] = None):
    """Evaluate model performance."""
    model.eval()
    all_y_trips = []
    all_y_hat_trips = []
    all_masks = []
    
    for x, (y_imf, y_trips) in loader:
        mask = None
        if subset_idx is not None and N_full is not None:
            x, _ = pad_batch_to_full(x, subset_idx, N_full)
            y_trips, mask = pad_batch_to_full(y_trips, subset_idx, N_full)
            all_masks.append(mask.cpu())
        else:
            all_masks.append(torch.ones_like(y_trips.cpu()))

        x, y_trips = to_device(x, device), to_device(y_trips, device)
        
        y_hat_imf = model(x)
        y_hat_trips = y_hat_imf.sum(dim=-1, keepdim=True)
        
        # Ensure tensors are moved to CPU properly
        if torch.is_tensor(y_trips):
            all_y_trips.append(y_trips.detach().cpu())
        else:
            all_y_trips.append(torch.tensor(y_trips).cpu())
        all_y_hat_trips.append(y_hat_trips.detach().cpu())

    # Concatenate all results
    all_y_trips = torch.cat(all_y_trips, dim=0)
    all_y_hat_trips = torch.cat(all_y_hat_trips, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate metrics on the full dataset
    from .metrics import _mae, _rmse, _mape, _mase, _r2
    mae = _mae(all_y_hat_trips, all_y_trips, mask=all_masks)
    rmse = _rmse(all_y_hat_trips, all_y_trips, mask=all_masks)
    mape = _mape(all_y_hat_trips, all_y_trips, mask=all_masks)
    mase = _mase(all_y_hat_trips, all_y_trips, mase_scaler, mask=all_masks)
    r2 = _r2(all_y_hat_trips, all_y_trips, mask=all_masks)
    
    return mae, rmse, mape, mase, r2, all_y_trips.numpy(), all_y_hat_trips.numpy()