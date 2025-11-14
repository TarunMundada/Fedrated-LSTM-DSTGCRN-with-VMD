"""
VMD (Variational Mode Decomposition) utilities.
"""
import numpy as np

# ================================================================
# Optional: VMD import (pip install vmdpy)
# ================================================================
try:
    from vmdpy import VMD
except Exception:
    VMD = None

def apply_vmd(signal: np.ndarray, K: int = 3, alpha=2000, tau=0., DC=0, init=1, tol=1e-7):
    """Apply VMD to a 1D signal, return K modes shaped [K, T]."""
    if VMD is None:
        raise ImportError("vmdpy not installed. Install with: pip install vmdpy")
    u, _, _ = VMD(signal, alpha, tau, K, DC, init, tol)
    return u  # [K, T]