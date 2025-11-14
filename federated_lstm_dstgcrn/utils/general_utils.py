"""
General utility functions.
"""
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union

def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_device(x, device):
    """Move tensors to specified device."""
    if isinstance(x, (list, tuple)):
        return [to_device(xx, device) for xx in x]
    return x.to(device)