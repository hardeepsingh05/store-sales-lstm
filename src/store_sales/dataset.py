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
    Dataset for creating sliding windows for a global store sales forecasting model.

    Each sample consists of historical features, future features, static categorical features,
    and the target sales values along with normalization parameters (location and scale).
    The dataset ensures that the time series for each store and product family is continuous on a daily basis,
    and it validates that the sliding windows are correctly constructed according to the specified context length,
    horizon, and stride.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing the time series data with columns for store number, product family, date, and relevant features.
        context_length (int): Number of historical time steps in each window.
        horizon (int): Number of future time steps to predict.
        hist_features (list[str]): List of historical feature column names.
        future_features (list[str]): List of future feature column names.
        static_features (list[str]): List of static categorical feature column names.
        target_end_max (pd.Timestamp | None): Maximum allowed end date for the target window.
        exact_target_start (pd.Timestamp | None): Exact start date for the target window.
        stride (int): Step size for sliding the window across the time series.
        require_target (bool): Whether the target values are required for each sample.
        groups (list[dict]): List of group metadata for each store and product family.
        samples (list[tuple[int, int]]): List of tuples indicating the start and end indices of each sliding window.
        metadata (list[dict]): List of metadata dictionaries for each sample, including dates and other relevant information.
    """

    def __init__(
        self,
        df: pd.DataFrame,
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
        """
        Initialize the StoreSalesWindowDataset with the given parameters.

        Args:
            df (pd.DataFrame): The input DataFrame containing the time series data with columns for store number, product family, date, and relevant features.
            context_length (int): Number of historical time steps in each window.
            horizon (int): Number of future time steps to predict.
            hist_features (list[str]): List of historical feature column names.
            future_features (list[str]): List of future feature column names.
            static_features (list[str]): List of static categorical feature column names.
            target_end_max (pd.Timestamp | None): Maximum allowed end date for the target window.
            exact_target_start (pd.Timestamp | None): Exact start date for the target window.
            stride (int): Step size for sliding the window across the time series.
            require_target (bool): Whether the target values are required for each sample.
            groups (list[dict]): List of group metadata for each store and product family.
            samples (list[tuple[int, int]]): List of tuples indicating the start and end indices of each sliding window.
            metadata (list[dict]): List of metadata dictionaries for each sample, including dates and other relevant information.

        Returns:
            None

        Raises:
            ValueError: If context_length, horizon, or stride are not greater than 0.
        """
        super().__init__()

        if context_length <= 0:
            raise ValueError(
                "context_length must be > 0"
            )

        if horizon <= 0:
            raise ValueError(
                "horizon must be > 0"
            )

        if stride <= 0:
            raise ValueError(
                "stride must be > 0"
            )

        self.context_length = context_length
        self.horizon = horizon

        self.hist_features = hist_features
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
        self.samples: list[
            tuple[int, int]
        ] = []

        self.metadata: list[dict] = []

        grouped = df.groupby(
            [
                "store_nbr",
                "family",
            ],
            sort=False,
        )

        for key, group in grouped:
            group = (
                group
                .sort_values("date")
                .reset_index(drop=True)
            )

            # ------------------------------------------------
            # Verify daily continuity
            # ------------------------------------------------

            date_diffs = (
                group["date"]
                .diff()
                .dropna()
            )

            if not (
                date_diffs
                == pd.Timedelta(days=1)
            ).all():
                raise ValueError(
                    "Non-daily gap found for "
                    f"store={key[0]}, "
                    f"family={key[1]}"
                )

            record = {
                "dates": (
                    group["date"]
                    .to_numpy()
                ),

                "hist": (
                    group[
                        hist_features
                    ]
                    .to_numpy(
                        dtype=np.float32
                    )
                ),

                "future": (
                    group[
                        future_features
                    ]
                    .to_numpy(
                        dtype=np.float32
                    )
                ),

                "target": (
                    group["sales_log"]
                    .to_numpy(
                        dtype=np.float32
                    )
                ),

                "static": (
                    group[
                        static_features
                    ]
                    .iloc[0]
                    .to_numpy(
                        dtype=np.int64
                    )
                ),

                "loc": float(
                    group[
                        "target_loc"
                    ].iloc[0]
                ),

                "scale": float(
                    group[
                        "target_scale"
                    ].iloc[0]
                ),
            }

            group_id = len(
                self.groups
            )

            self.groups.append(
                record
            )

            max_t = (
                len(group)
                - horizon
            )

            for t in range(
                context_length,
                max_t + 1,
                stride,
            ):
                target_start = pd.Timestamp(
                    record["dates"][t]
                )

                target_end = pd.Timestamp(
                    record["dates"][
                        t + horizon - 1
                    ]
                )

                # --------------------------------------------
                # Restrict training windows
                # --------------------------------------------

                if target_end_max is not None:
                    if (
                        target_end
                        > target_end_max
                    ):
                        continue

                # --------------------------------------------
                # Used for validation / inference
                # --------------------------------------------

                if (
                    exact_target_start
                    is not None
                ):
                    if (
                        target_start
                        != exact_target_start
                    ):
                        continue

                hist = record["hist"][
                    t
                    - context_length:
                    t
                ]

                future = record["future"][
                    t:
                    t + horizon
                ]

                if not np.isfinite(
                    hist
                ).all():
                    continue

                if not np.isfinite(
                    future
                ).all():
                    continue

                if require_target:
                    y = record["target"][
                        t:
                        t + horizon
                    ]

                    if not np.isfinite(
                        y
                    ).all():
                        continue

                self.samples.append(
                    (
                        group_id,
                        t,
                    )
                )

                self.metadata.append(
                    {
                        "store_nbr": key[0],
                        "family": key[1],
                        "target_start":
                            target_start,
                    }
                )

    def __len__(
        self,
    ) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return len(
            self.samples
        )

    def __getitem__(
        self,
        idx: int,
    ):
        """
        Retrieves a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            Tuple containing historical features, future features, static categorical features, target values, location, and scale tensors.
        """
        group_id, t = (
            self.samples[idx]
        )

        group = (
            self.groups[
                group_id
            ]
        )

        x_hist = (
            group["hist"][
                t
                - self.context_length:
                t
            ]
        )

        x_future = (
            group["future"][
                t:
                t + self.horizon
            ]
        )

        static_cat = (
            group["static"]
        )

        if self.require_target:
            y_log = (
                group["target"][
                    t:
                    t + self.horizon
                ]
            )
        else:
            y_log = np.zeros(
                self.horizon,
                dtype=np.float32,
            )

        x_hist = np.ascontiguousarray(
            x_hist
        )

        x_future = np.ascontiguousarray(
            x_future
        )

        static_cat = (
            np.ascontiguousarray(
                static_cat
            )
        )

        y_log = np.ascontiguousarray(
            y_log
        )

        return (
            torch.from_numpy(
                x_hist
            ),
            torch.from_numpy(
                x_future
            ),
            torch.from_numpy(
                static_cat
            ),
            torch.from_numpy(
                y_log
            ),
            torch.tensor(
                group["loc"],
                dtype=torch.float32,
            ),
            torch.tensor(
                group["scale"],
                dtype=torch.float32,
            ),
        )


def create_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    """
    Creates a DataLoader for the given dataset with specified parameters.

    Args:
        dataset (Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int): Random seed for shuffling.
        num_workers (int, optional): Number of subprocesses to use for data loading. Defaults to 0.

    Returns:
        DataLoader: Configured DataLoader instance.
    """
    generator = (
        torch.Generator()
    )

    generator.manual_seed(
        seed
    )

    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory":
            torch.cuda.is_available(),
        "drop_last": False,
    }

    if shuffle:
        kwargs["generator"] = (
            generator
        )

    if num_workers > 0:
        kwargs[
            "persistent_workers"
        ] = True

    return DataLoader(
        **kwargs
    )
