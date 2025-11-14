# Federated LSTM-DSTGCRN with VMD

## Overview

This repository contains the implementation of a Federated Learning approach for time series forecasting using a hybrid model that combines Variational Mode Decomposition (VMD), Graph Convolutional Recurrent Networks (GCRN), and Graph Attention Networks (GAT). The model is designed for transportation demand forecasting and can work with multiple data sources in a federated setting.

The core model, LSTM-DSTGCRN (Long Short-Term Memory - Dynamic Spatial-Temporal Graph Convolutional Recurrent Network), incorporates:
- **VMD preprocessing**: Decomposes time series data into Intrinsic Mode Functions (IMFs) for better feature extraction
- **Hybrid spatial modeling**: Combines GCRN and GAT for enhanced spatial feature learning
- **Federated Learning**: Enables distributed training across multiple clients without sharing raw data
- **Selective Integration**: Allows clients to selectively adopt global model components that improve their local performance

### Key Features
- Support for VMD preprocessing to decompose complex time series
- Hybrid GAT-GCRN spatial modeling with multiple fusion strategies
- Federated learning implementation with selective integration
- Comprehensive evaluation metrics (MAE, RMSE, MAPE, MASE, R²)
- Rich visualization capabilities for model analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Clone the Repository
```bash
git clone <repository-url>
cd "Fedrated-LSTM-DSTGCRN with VMD"
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Additional Dependencies
For VMD functionality, install the vmdpy package:
```bash
pip install vmdpy
```

## Usage

### Data Preparation
The model expects two CSV files for each dataset:
1. `tripdata_full.csv` - Contains timestamp and trip count data for each location
2. `weatherdata_full.csv` - Contains timestamp and weather data (temperature, precipitation) for each location

Example datasets are provided in the [DATA/TransportModes](file:///c%3A/Users/tarun/OneDrive/Desktop/Fedrated-LSTM-DSTGCRN%20with%20VMD/DATA/TransportModes) directory:
- [CHI-taxi](file:///c%3A/Users/tarun/OneDrive/Desktop/Fedrated-LSTM-DSTGCRN%20with%20VMD/DATA/TransportModes/CHI-taxi) - Chicago taxi demand data
- [NYC-bike](file:///c%3A/Users/tarun/OneDrive/Desktop/Fedrated-LSTM-DSTGCRN%20with%20VMD/DATA/TransportModes/NYC-bike) - New York City bike demand data
- [NYC-taxi](file:///c%3A/Users/tarun/OneDrive/Desktop/Fedrated-LSTM-DSTGCRN%20with%20VMD/DATA/TransportModes/NYC-taxi) - New York City taxi demand data

### Running Experiments

#### Using Batch Scripts (Windows)
Two batch scripts are provided for running experiments:

1. **VMD-GCRN Experiment** (GCRN-only spatial modeling):
   ```cmd
   run_vmd_expriment.bat
   ```

2. **VMD-GAT-Hybrid Experiment** (Hybrid GAT+GCRN spatial modeling):
   ```cmd
   run_gat_experiment.bat
   ```

#### Using Python Directly
You can also run experiments directly using Python:

```bash
python main.py --trip_csv "DATA/TransportModes/CHI-taxi/tripdata_full.csv" --dataset "CHI-taxi" --rounds 10 --clients 3 --vmd_k 3 --use_gat 1 --use_gcrn 1
```

### Command Line Arguments
- `--trip_csv`: Path to trip data CSV file
- `--dataset`: Dataset name for logging and saving results
- `--rounds`: Number of federated learning rounds
- `--clients`: Number of federated clients
- `--tin`: Input sequence length (default: 12)
- `--tout`: Output horizon (default: 3)
- `--vmd_k`: Number of VMD modes (0 to disable VMD)
- `--use_gat`: Enable GAT spatial modeling (1=enable, 0=disable)
- `--use_gcrn`: Enable GCRN spatial modeling (1=enable, 0=disable)
- `--gat_heads`: Number of GAT attention heads
- `--gat_fusion_mode`: Fusion mode for GAT-GCRN combination (concat, add, weighted)

## Repo Structure

```
.
├── main.py                           # Main entry point
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── run_gat_experiment.bat            # Batch script for GAT hybrid experiments
├── run_vmd_expriment.bat             # Batch script for VMD-GCRN experiments
├── DATA/
│   └── TransportModes/
│       ├── CHI-taxi/
│       │   ├── tripdata_full.csv
│       │   └── weatherdata_full.csv
│       ├── NYC-bike/
│       │   ├── tripdata_full.csv
│       │   └── weatherdata_full.csv
│       └── NYC-taxi/
│           ├── tripdata_full.csv
│           └── weatherdata_full.csv
└── federated_lstm_dstgcrn/           # Refactored code modules
    ├── __init__.py
    ├── experiment_runner.py          # Main experiment runner
    ├── models/
    │   ├── __init__.py
    │   ├── graph_components.py       # Adaptive adjacency and graph convolution
    │   ├── gat_components.py         # Graph attention network components
    │   ├── lstm_cells.py             # Graph convolutional LSTM cells
    │   └── lstm_dstgcrn.py           # Main LSTM-DSTGCRN model
    ├── data/
    │   ├── __init__.py
    │   ├── dataset.py                # TripWeatherDataset implementation
    │   └── utils.py                  # Data utilities (collate, padding)
    ├── training/
    │   ├── __init__.py
    │   ├── metrics.py                # Evaluation metrics (MAE, RMSE, etc.)
    │   └── trainer.py                # Training and evaluation functions
    ├── federated/
    │   ├── __init__.py
    │   ├── fl_components.py          # FLClient and FLServer classes
    │   └── selective_integration.py  # Selective integration functionality
    ├── visualization/
    │   ├── __init__.py
    │   └── plots.py                  # Visualization and plotting functions
    └── utils/
        ├── __init__.py
        ├── general_utils.py          # General utilities (seed, device handling)
        └── vmd_utils.py              # VMD utilities and apply_vmd function
```

### Output Files
Results are saved in the specified `save_dir` with the following files:
- `*_global_results.json`: Global metrics for each federated round
- `*_computational_metrics.json`: Performance and resource usage metrics
- `*_vmd_results.json`: VMD decomposition quality metrics (when VMD is enabled)
- Various plots:
  - Performance metrics over rounds
  - Training loss curves
  - Module replacement choices
  - Temporal stability analysis
  - True vs predicted values
  - Error distribution
  - Frequency domain analysis (when VMD is enabled)
  - VMD decomposition and spectra (when VMD is enabled)
  - GAT attention visualization (when GAT is enabled)