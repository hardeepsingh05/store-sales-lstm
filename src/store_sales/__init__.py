from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from torch.utils.data import (
    DataLoader,
    Dataset,
)

class StoreSalesWindowDataset(Dataset):
    """
    Creates sliding windows for global forecasting model.

    Each sample contains:

    x_hist: [context_length, historical_feature_count]
    x_future: [horizon, future_feature_count]
    static_cat: [categorical_feature_count]
    y_log: [horizon]
    target_loc: scalar
    target_scale: scalar
    """

    def __init__(
        self,
        df:pd.DataFrame,
        context_length: int,
        horizon: int,
        hist_features: list[str],
        future_features: list[str],
        static_features: list[str],
        target_end_max: pd.Timestamp | None = None,
        exact_target_start: pd.Timestamp | None = None,
        stride: int = 1,
        require_target: bool = True,
        ):

        super().__init__()

        if context_length <= 0:
            raise ValueError("context_length must be > 0")
        if horizon <=0:
            raise ValueError("horizon must be > 0")
        if stride <= 0 :
            raise ValueError("stride must be > 0")

        self.context_length = context_length
        self.horizon = horizon
        self.hist_featuers = hist_features
        self.future_features = future_features
        self.static_features = static_features
        self.require_target = require_target

        if target_end_max is not None:
            target_end_max = pd.Timestamp(
                target_end_max
            )
        if exact_target_start is not None:
            exact_target_start = pd.Timestamp(
                exact_target_start
            )
        
        self.groups: list[dict] = []
        self.sample: list[tuple[int, int]] = []
        self.metadata: list[dict] = []

        grouped = df.groupby(
            ["store_nbr", "family"],
            sort=False
        )

        for key,group in grouped:
            group = (group.sort_values("date").reset_index(drop=True))

            # -------------------------------
            # Verify daily continuity
            # ------------------------------
            date_diffs = (
                group["date"].diff().dropna()
            )

            if not (date_diffs == pd.Timedelta(days = 1)).all():
                raise ValueError(
                    "Non-daily gap found for f"
                    f"store={key[0]}, "
                    f"family={key[1]}"
                )
            
            record = {
                "dates": (group["date"].to_numpy()),
                "hist": (group[hist_features].to_numpy(dtype=np.float32)),
                "future": (group[future_features].to_numpy(dtype=np.float32)),
                "target": (group["sales_log"].to_numpy(dtype=np.float32)),
                "static": (group[static_features].iloc[0].to_numpy(dtype=np.int64)),
                "loc": float(group["target_loc"].iloc[0]),
                "scale": float(group["target_scale"].iloc[0]),
            }

            group_id = len(self.groups)
            self.groups.append(record)
            max_t = (len(group) - horizon)

            for t in range(
                context_length,
                max_t + 1,
                stride,
                ):
                target_start = pd.Timestamp(
                    record["dates"][t]
                )
                target_end = pd.Timestamp(
                    record["dates"][t + horizon - 1]
                )

                # ------------------------------
                # Restrict training windows
                # --------------------------------
                if target_end_max is not None:
                    if (target_end > target_end_max): 
                        continue

                # ------------------------------------------
                # Used for validation / inference
                # --------------------------------------------
                if (exact_target_start is not None):
                    if (target_start != exact_target_start):
                        continue
                
                hist = record["hist"][t - context_length: t]
                future = record["future"][t: t + horizon]

                if not np.isfinite(hist).all():
                    continue
                if not np.isfinite(future).all():
                    continue
                    
                if require_target:
                    y = record["target"][t: t + horizon]
                    if not np.isfinite(y).all():
                        continue

                self.samples.append(
                    (group_id, t,)
                )

                self.metadata.append(
                    {
                        "store_nbr": key[0],
                        "family": key[1],
                        "target_start": target_start,
                    }
                )
    
    def __len__(self,) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int,
    ):
        group_id, t = ( self.samples[idx])
        group = (self.groups[group_id])

        x_hist = (
            group["hist"][t - self.context_length: t]
        )
        x_future = (
            group["future"][t: t + self.horizon]
        )
        static_cat = (
            group["static"]
        )
        
        if self.require_target:
            y_log = (group["target"][t: t + self.horizon])
        else:
            y_log = np.zeroes(
                self.horizon,
                dtype=np.float32,
            )
        
        x_hist = np.ascontiguousarray(x_hist)
        x_future = np.ascontiguousarray(x_future)
        static_cat = (np.ascontiguousarray(static_cat))

        y_log = np.ascontiguousarray(y_log)

        return (
            torch.from_numpy(x_hist),
            torch.from_numpy(x_future),
            torch.from_numpy(static_cat),
            torch.from_numpy(y_log),
            torch.tensor(group["loc"], dtype=torch.float32,),
            torch.tensor(group["scale"], dtype=torch.float32,),
        )

def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = (
        torch.Generator()
    )
    generator.manual_seed(seed)

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": False,
    }

    if shuffle:
        kwargs["generator"] = (generator)
    
    if num_workers > 0:
        kwargs["persistent_workers"] = True
    
    return DataLoader(**kwargs)
    
