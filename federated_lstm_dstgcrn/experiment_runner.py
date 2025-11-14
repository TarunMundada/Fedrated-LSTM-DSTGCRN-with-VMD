"""
Main experiment runner for federated LSTM-DSTGCRN with VMD.
"""
import os
import copy
import json
import argparse
import tracemalloc
import time
import numpy as np
import torch
import pandas as pd
import torch.utils.data
from typing import List, Optional, Dict, Tuple
from .utils.general_utils import set_seed
from .utils.vmd_utils import apply_vmd
from .models.lstm_dstgcrn import LSTMDSTGCRN
from .data.dataset import TripWeatherDataset
from .federated.fl_components import FLClient, FLServer, ClientConfig
from .visualization.plots import (
    _plot_all_metrics, _plot_training_loss, _plot_module_replacement,
    _plot_temporal_stability, _plot_true_vs_predicted, _plot_error_distribution,
    _plot_comprehensive_summary, _plot_frequency_domain_analysis,
    _plot_gat_attention, _plot_gat_vs_gcrn_comparison,
    _plot_vmd_decomposition, _plot_vmd_imf_spectra, _calculate_snr
)

def run_federated(
    trip_csv: str,
    dataset_name: str,
    num_clients: int = 3,
    num_rounds: int = 10,
    input_len: int = 12,
    output_len: int = 3,
    device: Optional[str] = None,
    save_dir: str = "./ckpts",
    epochs_per_round: int = 2,
    batch_size: int = 16,
    use_attention: bool = False,
    attn_heads: int = 2,
    vmd_k: int = 0,
    save_vmd: bool = True,
    ablate_adja: bool = False,
    # GAT parameters
    use_gat: bool = False,
    use_gcrn: bool = True,
    gat_heads: int = 4,
    gat_dropout: float = 0.1,
    gat_fusion_mode: str = 'concat',
):
    # --- Setup ---
    os.makedirs(save_dir, exist_ok=True)
    set_seed(42)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Memory optimization for complex models
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB available")
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.8)
    
    # Set thread count for CPU efficiency
    torch.set_num_threads(4)
    
    tracemalloc.start()
    start_time = time.time()

    # --- Infer N (nodes) ---
    tmp = pd.read_csv(trip_csv, nrows=1)
    node_cols = [c for c in tmp.columns if c != "timestamp"]
    N_full = len(node_cols)
    node_indices = list(range(N_full))

    # --- VMD Analysis ---
    vmd_results = {}
    vmd_reconstruction_for_plot = None
    original_signal_for_plot = None
    if vmd_k > 0:
        print("\n--- VMD analysis ---")
        tmp_ds = TripWeatherDataset(trip_csv, vmd_k=0, node_subset=node_indices[:1]) # just for one node
        signal = tmp_ds.full_trip_series[:, 0]
        imfs = apply_vmd(signal, K=vmd_k)
        reconstructed = imfs.sum(axis=0)
        
        reconstruction_snr = _calculate_snr(signal, signal - reconstructed)
        
        vmd_results = {"Reconstruction_SNR_dB": reconstruction_snr}
        print(f"VMD Reconstruction SNR: {reconstruction_snr:.2f} dB")
        json_path_vmd = os.path.join(save_dir, f"{dataset_name}_vmd_results.json")
        with open(json_path_vmd, "w") as f:
            json.dump(vmd_results, f, indent=2)
            
        original_signal_for_plot = signal
        vmd_reconstruction_for_plot = reconstructed


    # --- Client and Server Setup ---
    chunks = [node_indices[i::num_clients] for i in range(num_clients)]
    input_dim = (vmd_k if vmd_k > 0 else 1) + 5
    output_dim = (vmd_k if vmd_k > 0 else 1)
    
    def model_fn():
        return LSTMDSTGCRN(
            num_nodes=N_full, 
            input_dim=input_dim, 
            hidden_dim=32,  # Reduced from 64
            output_dim=output_dim,
            horizon=output_len, 
            use_attention=use_attention, 
            num_heads=attn_heads, 
            ablate_adja=ablate_adja,
            # GAT parameters
            use_gat=use_gat,
            use_gcrn=use_gcrn,
            gat_heads=gat_heads,
            gat_dropout=gat_dropout,
            gat_fusion_mode=gat_fusion_mode,
        )

    clients: List[FLClient] = []
    print("\n--- Initializing Datasets for Clients ---")
    for cid, subset in enumerate(chunks):
        full_ds = TripWeatherDataset(trip_csv, input_len, output_len, stride=1, node_subset=subset, vmd_k=vmd_k, save_vmd=save_vmd and cid == 0)
        n_total = len(full_ds)
        if n_total < 10: # Ensure enough data for split
            print(f"Warning: Client {cid} has only {n_total} samples. Skipping this client.")
            continue
        n_tr = int(0.70 * n_total)
        n_va = int(0.15 * n_total)
        n_te = n_total - n_tr - n_va
        tr, va, te = torch.utils.data.random_split(full_ds, [n_tr, n_va, n_te], generator=torch.Generator().manual_seed(123 + cid))
        cfg = ClientConfig(id=cid, epochs=epochs_per_round, batch_size=batch_size, lr=1e-3)
        clients.append(FLClient(cid, model_fn, tr, va, te, cfg, device, subset_idx=subset, N_full=N_full))

    if not clients:
        raise ValueError("No clients were created. Check data size and splitting logic.")
        
    server = FLServer(model_fn, clients, N_full=N_full)
    global_log = []

    # --- Federated Training Loop ---
    for r in range(num_rounds):
        print(f"\n===== Round {r+1}/{num_rounds} | {dataset_name} =====")
        round_start_time = time.time()
        
        print("Distributing global model to clients...")
        server.distribute()
        
        client_train_losses = []
        best_groups_this_round = []
        
        for c_idx, c in enumerate(clients):
            print(f"Client {c_idx + 1}/{len(clients)}: Starting integration...")
            best_group = c.integrate(server.global_model)
            best_groups_this_round.append(best_group)
            
            print(f"Client {c_idx + 1}/{len(clients)}: Starting local training...")
            local_losses = c.train_local()
            client_train_losses.append(np.mean(local_losses))
            print(f"Client {c_idx + 1}/{len(clients)}: Training completed with loss {np.mean(local_losses):.4f}")
        
        avg_client_loss = np.mean(client_train_losses)

        # Weighted global training metrics (on each client's TRAIN set)
        train_metrics = [(c.train_metrics()[:5], c.n_train()) for c in clients]
        total_train = sum(n for _, n in train_metrics)
        global_train_mae = sum(m[0] * n for (m), n in train_metrics) / total_train
        global_train_rmse = sum(m[1] * n for (m), n in train_metrics) / total_train

        # Evaluate test metrics per client and aggregate
        test_metrics_data = [c.test_metrics() for c in clients]
        total_test = sum(c.n_test() for c in clients)
        
        global_mae = sum(m[0] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_rmse = sum(m[1] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_mape = sum(m[2] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_mase = sum(m[3] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        global_r2 = sum(m[4] * c.n_test() for m, c in zip(test_metrics_data, clients)) / total_test
        
        print(f"Global Test MAE: {global_mae:.4f} | RMSE: {global_rmse:.4f} | R^2: {global_r2:.4f}")
        print(f"Avg Train Loss: {avg_client_loss:.4f} | Global Train MAE: {global_train_mae:.4f}")

        log_entry = {
            "round": r + 1, "global_train_loss_avg": float(avg_client_loss),
            "global_train_mae": float(global_train_mae), "global_train_rmse": float(global_train_rmse),
            "global_test_mae": float(global_mae), "global_test_rmse": float(global_rmse),
            "global_test_mape": float(global_mape), "global_test_mase": float(global_mase), "global_test_r2": float(global_r2),
            "module_replacement": best_groups_this_round
        }
        global_log.append(log_entry)

        states = [(c.state_dict(), c.n_train()) for c in clients]
        server.aggregate_fedavg(states)
        
        print(f"Round {r+1} took {time.time() - round_start_time:.2f} seconds.")
    
    # --- Final Evaluation and Reporting ---
    print("\n--- Final Evaluation and Plotting ---")
    server.distribute()
    
    final_y_true_list, final_y_pred_list = [], []
    for c in clients:
        _, _, _, _, _, y_true, y_pred = c.test_metrics()
        final_y_true_list.append(y_true)
        final_y_pred_list.append(y_pred)
    
    final_y_true = np.concatenate(final_y_true_list, axis=0)
    final_y_pred = np.concatenate(final_y_pred_list, axis=0)

    # Save results
    json_path = os.path.join(save_dir, f"{dataset_name}_global_results.json")
    with open(json_path, "w") as f:
        json.dump(global_log, f, indent=2)
    print(f"Saved global results to {json_path}")

    # Plotting
    module_names = list(getattr(clients[0].model, 'modules_to_integrate', {}).keys())
    _plot_all_metrics(global_log, save_dir, dataset_name)
    _plot_training_loss(global_log, save_dir, dataset_name)
    _plot_module_replacement(global_log, save_dir, dataset_name, module_names)
    _plot_temporal_stability(final_y_true, final_y_pred, save_dir, dataset_name)
    _plot_true_vs_predicted(final_y_true, final_y_pred, save_dir, dataset_name)
    _plot_error_distribution(final_y_true, final_y_pred, save_dir, dataset_name)
    
    # GAT-related plots
    if use_gat:
        _plot_gat_attention(save_dir, dataset_name)  # Use the existing function with correct name
        _plot_gat_vs_gcrn_comparison(global_log, save_dir, dataset_name)
    
    # VMD-related plots
    if vmd_k > 0 and original_signal_for_plot is not None:
        # Re-evaluate final model on first client's test set to get preds for frequency plot
        # The returned arrays will have shape [samples, horizon, nodes_in_client, features]
        _, _, _, _, _, y_true_client, y_pred_client = clients[0].test_metrics()
        
        # Select ONLY the first node for this client for a 1-to-1 comparison
        y_true_for_freq = y_true_client[:, :, 0, :].flatten() # Select first node and flatten
        y_pred_for_freq = y_pred_client[:, :, 0, :].flatten() # Select first node and flatten

        _plot_frequency_domain_analysis(
            y_true_for_freq, 
            vmd_reconstruction_for_plot[:len(y_true_for_freq)] if vmd_reconstruction_for_plot is not None else None, 
            y_pred_for_freq, 
            save_dir, 
            dataset_name
        )
        
        # Add VMD-specific plots
        try:
            # Get a sample signal and its IMFs for plotting
            tmp_ds = TripWeatherDataset(trip_csv, vmd_k=0, node_subset=[0]) # just for one node
            sample_signal = tmp_ds.full_trip_series[:, 0]
            sample_imfs = apply_vmd(sample_signal, K=vmd_k)
            
            _plot_vmd_decomposition(sample_signal, sample_imfs, save_dir, dataset_name, node_id=0)
            _plot_vmd_imf_spectra(sample_imfs, save_dir, dataset_name, node_id=0)
        except Exception as e:
            print(f"Warning: Could not generate VMD analysis plots: {e}")

    # --- Final Tables ---
    final_metrics = {
        "Dataset": dataset_name,
        "MAE": global_log[-1]["global_test_mae"] if global_log else float('nan'),
        "RMSE": global_log[-1]["global_test_rmse"] if global_log else float('nan'),
        "MAPE (%)": global_log[-1]["global_test_mape"] if global_log else float('nan'),
        "MASE": global_log[-1]["global_test_mase"] if global_log else float('nan'),
        "R^2": global_log[-1]["global_test_r2"] if global_log else float('nan'),
    }
    
    try:
        print("\n--- Performance Comparison ---")
        print(pd.DataFrame([final_metrics]).set_index("Dataset").to_markdown(floatfmt=".4f"))
    except Exception as e:
        print(f"\nError generating performance metrics table: {e}")
        # Fallback to basic print
        if global_log:
            print(f"\nFinal Performance Metrics:")
            print(f"  MAE: {global_log[-1]['global_test_mae']:.4f}")
            print(f"  RMSE: {global_log[-1]['global_test_rmse']:.4f}")
            print(f"  MAPE: {global_log[-1]['global_test_mape']:.4f}%")
            print(f"  MASE: {global_log[-1]['global_test_mase']:.4f}")
            print(f"  R^2: {global_log[-1]['global_test_r2']:.4f}")
    
    total_runtime = time.time() - start_time
    try:
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        runtime_and_memory = {
            "Configuration": dataset_name,
            "Total Runtime (s)": total_runtime,
            "Peak Memory (MB)": peak_mem / 1024 / 1024,
            "VMD (K)": vmd_k,
            "Attention": "Yes" if use_attention else "No",
            "Adaptive Adjacency": "No (Ablated)" if ablate_adja else "Yes",
            "GAT": "Yes" if use_gat else "No",
            "GCRN": "Yes" if use_gcrn else "No",
        }
        print("\n--- Computational Metrics ---")
        print(pd.DataFrame([runtime_and_memory]).set_index("Configuration").to_markdown(floatfmt=".2f"))
        
        # Save computational metrics to JSON file
        comp_metrics_path = os.path.join(save_dir, f"{dataset_name}_computational_metrics.json")
        with open(comp_metrics_path, "w") as f:
            json.dump({
                "performance_metrics": final_metrics,
                "computational_metrics": runtime_and_memory
            }, f, indent=2)
        print(f"\nSaved computational metrics to {comp_metrics_path}")
    except Exception as e:
        print(f"\nError generating computational metrics: {e}")
        print(f"\nBasic Computational Info:")
        print(f"  Total Runtime: {total_runtime:.2f} seconds")
        print(f"  VMD (K): {vmd_k}")
        print(f"  Attention: {'Yes' if use_attention else 'No'}")
        print(f"  Adaptive Adjacency: {'No (Ablated)' if ablate_adja else 'Yes'}")
        print(f"  GAT: {'Yes' if use_gat else 'No'}")
        print(f"  GCRN: {'Yes' if use_gcrn else 'No'}")

    # Generate additional summary plots
    try:
        _plot_comprehensive_summary(global_log, save_dir, dataset_name, final_metrics)
    except Exception as e:
        print(f"\nWarning: Could not generate comprehensive summary plot: {e}")

    print("\n--- Experiment Completed Successfully ---")
    print(f"Results saved to: {save_dir}")

def main():
    """Main entry point for the experiment runner."""
    ap = argparse.ArgumentParser(description="Federated Time Series Forecasting with LSTM-DSTGCRN")
    ap.add_argument("--trip_csv", required=True, help="Path to *tripdata*_full.csv (e.g., 'CHI-taxi/tripdata_full.csv')")
    ap.add_argument("--dataset", default="Dataset", help="Name for the dataset for titles and filenames (e.g., 'NYC-Bike')")
    ap.add_argument("--rounds", type=int, default=10, help="Number of federated rounds")
    ap.add_argument("--clients", type=int, default=3, help="Number of clients")
    ap.add_argument("--tin", type=int, default=12, help="Length of input sequence")
    ap.add_argument("--tout", type=int, default=3, help="Length of output horizon")
    ap.add_argument("--epochs_per_round", type=int, default=2, help="Local epochs per client per round")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--save_dir", default="./results", help="Directory to save checkpoints, logs, and plots")
    # --- Model Configuration ---
    ap.add_argument("--vmd_k", type=int, default=0, help="If >0, use VMD with K modes. Model predicts K IMFs and reconstructs for loss/metrics")
    ap.add_argument("--use_attention", type=int, default=0, help="Use 1 to enable multi-head attention, 0 to disable")
    ap.add_argument("--attn_heads", type=int, default=2, help="Number of attention heads")
    # --- GAT Configuration ---
    ap.add_argument("--use_gat", type=int, default=0, help="Use 1 to enable GAT spatial modeling, 0 to disable")
    ap.add_argument("--use_gcrn", type=int, default=1, help="Use 1 to enable GCRN spatial modeling, 0 to disable")
    ap.add_argument("--gat_heads", type=int, default=4, help="Number of GAT attention heads")
    ap.add_argument("--gat_dropout", type=float, default=0.1, help="GAT dropout rate")
    ap.add_argument("--gat_fusion_mode", default="concat", choices=["concat", "add", "weighted"], 
                    help="Fusion mode for combining GCRN and GAT outputs: concat, add, or weighted")
    # --- Ablation ---
    ap.add_argument("--ablate_adja", type=int, default=0, help="Use 1 to replace adaptive adjacency with identity matrix for ablation study")
    # --- Other ---
    ap.add_argument("--save_vmd", type=int, default=1, help="Save VMD-preprocessed file once (client 0)")
    args = ap.parse_args()

    run_federated(
        trip_csv=args.trip_csv,
        dataset_name=args.dataset,
        num_clients=args.clients,
        num_rounds=args.rounds,
        input_len=args.tin,
        output_len=args.tout,
        epochs_per_round=args.epochs_per_round,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        use_attention=bool(args.use_attention),
        attn_heads=args.attn_heads,
        vmd_k=args.vmd_k,
        save_vmd=bool(args.save_vmd),
        ablate_adja=bool(args.ablate_adja),
        # GAT parameters
        use_gat=bool(args.use_gat),
        use_gcrn=bool(args.use_gcrn),
        gat_heads=args.gat_heads,
        gat_dropout=args.gat_dropout,
        gat_fusion_mode=args.gat_fusion_mode,
    )

if __name__ == "__main__":
    main()