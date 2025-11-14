"""
Federated learning components: Client and Server classes.
"""
import copy
import time
import torch
import numpy as np
from dataclasses import dataclass
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from ..data.utils import collate_batch
from ..training.trainer import train_one_epoch, evaluate
from ..training.metrics import calculate_mase_scaling_factor
from .selective_integration import selective_integration

@dataclass
class ClientConfig:
    id: int
    epochs: int = 2
    batch_size: int = 32
    lr: float = 1e-3

class FLClient:
    def __init__(self, cid, model_fn, train_ds, val_ds, test_ds, cfg: ClientConfig, device, subset_idx: List[int], N_full: int):
        self.cid = cid
        self.device = device
        self.cfg = cfg
        self.model = model_fn().to(device)
        self.train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_batch)
        self.val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
        self.test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_batch)
        self.subset_idx = subset_idx
        self.N_full = N_full
        # Calculate MASE scaler once from this client's training data
        self.mase_scaler = calculate_mase_scaling_factor(train_ds)

    def set_params_from(self, global_model):
        self.model.load_state_dict(global_model.state_dict())

    def integrate(self, global_model):
        self.model, best_group = selective_integration(self.model, global_model, self.val_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)
        return best_group

    def train_local(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        local_loss_history = []
        
        for epoch in range(self.cfg.epochs):
            print(f"  Client {self.cid} - Epoch {epoch + 1}/{self.cfg.epochs}")
            epoch_start = time.time()
            
            loss = train_one_epoch(self.model, self.train_loader, opt, self.device, subset_idx=self.subset_idx, N_full=self.N_full)
            local_loss_history.append(loss)
            
            epoch_time = time.time() - epoch_start
            print(f"  Client {self.cid} - Epoch {epoch + 1} completed in {epoch_time:.2f}s, loss: {loss:.4f}")
            
            # Timeout protection - if epoch takes too long, skip remaining
            if epoch_time > 300:  # 5 minutes timeout
                print(f"  Client {self.cid} - Epoch timeout, stopping early")
                break
                
        return local_loss_history

    def state_dict(self):
        return copy.deepcopy(self.model.state_dict())

    def n_train(self): 
        try:
            return len(self.train_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            return 0
    
    def n_test(self):
        try:
            return len(self.test_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            return 0
    
    def n_val(self): 
        try:
            return len(self.val_loader.dataset)  # type: ignore
        except (TypeError, AttributeError):
            return 0

    @torch.no_grad()
    def test_metrics(self):
        return evaluate(self.model, self.test_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

    @torch.no_grad()
    def train_metrics(self):
        return evaluate(self.model, self.train_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

    @torch.no_grad()
    def val_metrics(self):
        return evaluate(self.model, self.val_loader, self.device, self.mase_scaler, subset_idx=self.subset_idx, N_full=self.N_full)

class FLServer:
    def __init__(self, model_fn, clients: List[FLClient], N_full: int):
        self.global_model = model_fn()
        self.clients = clients
        self.N_full = N_full

    def distribute(self):
        for c in self.clients:
            c.set_params_from(self.global_model)

    def aggregate_fedavg(self, states: List[Tuple[Dict, int]]):
        total = sum(n for _, n in states)
        if total == 0: return
        new_sd = copy.deepcopy(states[0][0])
        for k in new_sd.keys():
            new_sd[k] = sum(sd[k] * (n/total) for sd, n in states)
        self.global_model.load_state_dict(new_sd)