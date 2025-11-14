"""
Evaluation metrics for model performance.
"""
import numpy as np
import torch
from sklearn.metrics import r2_score
from typing import Tuple, Optional

def _mae(a, b, mask=None):
    """Mean Absolute Error"""
    if mask is None:
        return (a - b).abs().mean()
    diff = (a - b).abs() * mask
    return diff.sum() / mask.sum().clamp(min=1)

def _rmse(a, b, mask=None):
    """Root Mean Square Error"""
    if mask is None:
        return torch.sqrt(((a - b) ** 2).mean())
    diff2 = ((a - b) ** 2) * mask
    return torch.sqrt(diff2.sum() / mask.sum().clamp(min=1))

def _mape(a, b, mask=None, epsilon=1e-8):
    """Mean Absolute Percentage Error"""
    abs_err = (a - b).abs()
    abs_true = b.abs()
    
    # Calculate MAPE
    per_err = abs_err / abs_true.clamp(min=epsilon)
    
    if mask is not None:
        masked_per_err = per_err * mask
        return 100. * (masked_per_err.sum() / mask.sum().clamp(min=1))
    else:
        return 100. * per_err.mean()

def _mase(a, b, scale_factor, mask=None):
    """Mean Absolute Scaled Error"""
    if scale_factor < 1e-8: return float('inf')
    mae_val = _mae(a, b, mask)
    return mae_val / scale_factor

def _r2(a, b, mask=None):
    """R-squared"""
    if mask is not None:
        mask_flat = mask.squeeze(-1).flatten().cpu().numpy() > 0.5
        a_masked = a.flatten().cpu().numpy()[mask_flat]
        b_masked = b.flatten().cpu().numpy()[mask_flat]
        if len(a_masked) < 2: # R2 is not well-defined for less than 2 points
            return 0.0
        return r2_score(b_masked, a_masked)
    return r2_score(b.cpu().numpy().flatten(), a.cpu().numpy().flatten())

def calculate_mase_scaling_factor(train_dataset) -> float:
    """Calculates the MASE scaling factor from the training set (naive forecast error)."""
    # Handle both Subset and regular Dataset objects
    if hasattr(train_dataset, 'dataset'):
        # This is a Subset object from random_split
        full_series = train_dataset.dataset.full_trip_series
        train_indices = train_dataset.indices
        # Get the part of the series corresponding to the training set
        train_series = full_series[min(train_indices):max(train_indices)+1]
    else:
        # This is a regular TripWeatherDataset
        full_series = train_dataset.full_trip_series
        train_series = full_series
    
    # Naive forecast (y_t = y_{t-1})
    naive_forecast_error = np.abs(train_series[1:] - train_series[:-1])
    return float(np.mean(naive_forecast_error))