"""
Dataset module for trip and weather data with optional VMD preprocessing.
"""
import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from typing import List, Optional, Tuple
from ..utils.vmd_utils import apply_vmd

class TripWeatherDataset(Dataset):
    """
    Builds features per the paper. Two modes:
      - VMD disabled (vmd_k=0): features = [trips, temperature, precipitation, hour_norm, day_norm, weekend]
      - VMD enabled (vmd_k=K):  features = [IMF_1..IMF_K, temperature, precipitation, hour_norm, day_norm, weekend]

    Targets always return both:
      - y_imfs: IMF stack (when K>0) or trips (when K=0)  -> used by model output
      - y_trips: raw trips (for reconstruction loss & metrics)
    """
    def __init__(
        self,
        trip_csv: str,
        input_len: int = 12,
        output_len: int = 3,
        stride: int = 1,
        node_subset: Optional[List[int]] = None,
        vmd_k: int = 0,
        save_vmd: bool = True,
    ):
        self.T_in = input_len
        self.T_out = output_len
        self.stride = stride
        self.node_subset = None if node_subset is None else list(node_subset)
        self.vmd_k = int(vmd_k)

        trips = pd.read_csv(trip_csv)
        weather_csv = trip_csv.replace("tripdata", "weatherdata")
        weathers = pd.read_csv(weather_csv)

        # parse timestamps & drop tz
        trips["timestamp"] = pd.to_datetime(trips["timestamp"]).dt.tz_localize(None)
        weathers["timestamp"] = pd.to_datetime(weathers["timestamp"]).dt.tz_localize(None)

        trips = trips.set_index("timestamp")
        weathers = weathers.set_index("timestamp")

        trips_np = trips.to_numpy()            # [T, N]
        weathers_np = weathers.to_numpy()      # [T, 2N]
        self.full_trip_series = trips_np # Store for MASE calculation

        # ---------- Time/aux features ----------
        # Ensure we have a proper DatetimeIndex for datetime operations
        try:
            # Try direct access first (most pandas versions)
            weekends = trips.index.dayofweek.isin([5, 6]).astype(int)  # type: ignore
            day_values = trips.index.dayofweek.to_numpy()  # type: ignore
            hour_values = trips.index.hour.to_numpy()  # type: ignore
        except AttributeError:
            # Fallback: convert to DatetimeIndex
            datetime_index = pd.to_datetime(trips.index)
            weekends = datetime_index.dayofweek.isin([5, 6]).astype(int)  # type: ignore
            day_values = datetime_index.dayofweek.to_numpy()  # type: ignore
            hour_values = datetime_index.hour.to_numpy()  # type: ignore
        enc = OneHotEncoder(sparse_output=False)
        weekend_1hot = enc.fit_transform(weekends.reshape(-1, 1))[:, 0].reshape(-1, 1)
        weekend_1hot = np.repeat(weekend_1hot[:, np.newaxis, :], trips_np.shape[1], axis=1)  # [T,N,1]
        day_norm = (day_values / 6.0) # Monday=0, Sunday=6
        day_norm = np.repeat(day_norm[:, None], trips_np.shape[1], axis=1)
        hour_norm = (hour_values / 23.0) # 0 to 23
        hour_norm = np.repeat(hour_norm[:, None], trips_np.shape[1], axis=1)
        temperature = weathers_np[:, ::2]
        precipitation = weathers_np[:, 1::2]

        # ---------- Primary feature block (trips vs VMD IMFs) ----------
        if self.vmd_k > 0:
            # VMD decomposition per node
            imf_list = []
            print(f"Applying VMD (K={self.vmd_k}) to {trips_np.shape[1]} nodes...")
            for j in range(trips_np.shape[1]):
                modes = apply_vmd(trips_np[:, j], K=self.vmd_k)  # [K, T]
                imf_list.append(modes)
            imfs_stacked = np.stack(imf_list, axis=1).transpose(2, 1, 0)  # [T, N, K]

            # Save once per file (avoid duplicates across clients)
            vmd_path = trip_csv.replace("tripdata", f"tripdata_vmd{self.vmd_k}")
            if save_vmd and not os.path.exists(vmd_path):
                vmd_df = pd.DataFrame(imfs_stacked.reshape(imfs_stacked.shape[0], -1))
                vmd_df.to_csv(vmd_path, index=False)
                print(f"Saved VMD-preprocessed trips to {vmd_path}")

            primary = imfs_stacked  # [T,N,K]
            y_imfs_full = imfs_stacked  # [T,N,K]
        else:
            primary = trips_np[:, :, None]  # [T,N,1]
            y_imfs_full = primary  # treat trips as single-channel output

        # Build full feature tensor
        data = np.concatenate(
            (
                primary,
                temperature[:, :, None],
                precipitation[:, :, None],
                hour_norm[:, :, None],
                day_norm[:, :, None],
                weekend_1hot,
            ),
            axis=2,
        )  # [T,N, (K or 1) + 5]

        # Optionally subset nodes for this client
        if self.node_subset is not None:
            data_sub = data[:, self.node_subset, :]
            y_imfs_sub = y_imfs_full[:, self.node_subset, :]
            trips_sub = trips_np[:, self.node_subset][:, :, None]
        else:
            data_sub = data
            y_imfs_sub = y_imfs_full
            trips_sub = trips_np[:, :, None]

        self.data = torch.tensor(data_sub, dtype=torch.float32)      # [T,N,F]
        self.y_imfs = torch.tensor(y_imfs_sub, dtype=torch.float32)     # [T,N,K or 1]
        self.y_trips = torch.tensor(trips_sub, dtype=torch.float32)      # [T,N,1]

        # window indices
        Ttot = self.data.shape[0]
        self.windows = []
        for s in range(0, Ttot - (self.T_in + self.T_out) + 1, self.stride):
            x = self.data[s: s + self.T_in]                      # [T_in,N,F]
            y_imf = self.y_imfs[s + self.T_in: s + self.T_in + self.T_out]  # [T_out,N,K or 1]
            y_trip = self.y_trips[s + self.T_in: s + self.T_in + self.T_out] # [T_out,N,1]
            self.windows.append((x, y_imf, y_trip))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]