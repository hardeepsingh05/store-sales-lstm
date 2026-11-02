# Prediction utilities for the store sales LSTM model, including functions to generate predictions from a dataloader, convert predictions to a DataFrame, create submission files, and load models from checkpoints.

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from store_sales.config import (
    Config,
)
from store_sales.data import (
    load_raw_data, 
    validate_raw_data,
)
from store_sales.dataset import (
    StoreSalesWindowsDataset,
    create_dataloader,
)

from store_sales.features import (
    FUTURE_FEATURES,
    HIST_FEATURES,
    STATIC_CAT_FEATURES,
    build_feature_panel,
    get_temporal_split,
    load_preprocessor,
    transform_panel,
    validate_model_frame,
)

from store_sales.model import (
    GlobalSalesLSTM,
)

@torch.inference_mode()
def predict_loader(
    model,
    loader,
    device: torch.device,
) -> np.ndarray:
    """
    Generate predictions from a dataloader using the provided model and device.

    Args:
        model: The trained LSTM model for store sales prediction.
        loader: DataLoader providing batches of input data.
        device: The device to perform inference on (CPU or GPU).

    Returns:
        np.ndarray: Array of predicted sales values.

    Raises:
        ValueError: If the dataloader contains no samples.
    """
    model.eval()

    predictions = []
    for batch in loader:
        (
            x_hist,
            x_future,
            static_cat,
            _,
            target_loc,
            target_scale,
        ) = batch

        x_hist = (x_hist.to(device, non_blocking=True,))
        x_future = (x_future.to(device, non_blocking=True,))
        static_cat = (static_cat.to(device, non_blocking=True,))
        target_loc = (target_loc.to(device, non_blocking=True,).unsqueeze(1))
        target_scale = (target_scale.to(device, non_blocking=True,).unsqueeze(1))

        prediction_z = model(
            x_hist,
            x_future,
            static_cat,
        )

        prediction_log = (
            prediction_z * target_scale + target_loc
        )

        prediction_log = (
            torch.clamp(
                prediction_log,
                min=0.0
            )
        )

        prediction_sales = (
            torch.expm1(prediction_log)
        )
        predictions.append(
            prediction_sales.cpu().numpy()
        )

    if not predictions:
        raise ValueError("Prediction loader contains no samples.")

    return np.concatenate(
        predictions,
        axis=0,
    )

def predictions_to_frame(
    dataset: StoreSalesWindowsDataset,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Convert model predictions into a structured DataFrame with store, family, date, and horizon information.

    Args:
        dataset: The dataset containing metadata for each prediction.
        predictions: Array of predicted sales values.

    Returns:
        pd.DataFrame: DataFrame containing predictions with associated metadata.
    
    Raises:
        ValueError: If the number of predictions does not match the dataset metadata.
    """
    if (len(dataset.metadata) != predictions.shape[0]):
        raise ValueError("Prediction count does not match dataset metadata.")
    
    rows = []
    horizon = (
        predictions.shape[1]
    )

    for i, metadata in enumerate(dataset.metadata):
        dates = pd.date_range(
            metadata["target_start"],
            periods=horizon,
            freq="D",
        )
        for h, date in enumerate(dates):
            rows.append(
                {
                    "store_nbr":metadata["store_nbr"],
                    "family":metadata["family"],
                    "date": date,
                    "horizon": h + 1,
                    "prediction": max(0.0, float(predictions[i,h,]),),
                }
            )
    return pd.DataFrame(rows)


def create_submission(
    test: pd.DataFrame,
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a submission DataFrame by merging test data with model predictions.

    Args:
        test: DataFrame containing test data with columns 'id', 'date', 'store_nbr', and 'family'.
        prediction_df: DataFrame containing model predictions with columns 'store_nbr', 'family', 'date', and 'prediction'.

    Returns:
        pd.DataFrame: Submission DataFrame with columns 'id' and 'sales'.

    Raises:
        RuntimeError: If the submission row count differs from the test data.
        RuntimeError: If the submission contains missing predictions.
        RuntimeError: If the submission contains negative sales.
    """
    forecast = (
        prediction_df[
            [
                "store_nbr",
                "family",
                "date",
                "prediction",
            ]
        ].rename(
            columns={
                "prediction": "sales",
            }
        )
    )

    submission = (
        test[
            [
                "id",
                "date",
                "store_nbr",
                "family",
            ]
        ].merge(
            forecast, 
            on = [
                "date",
                "store_nbr",
                "family",
            ],
            how="left",
            validate="one-to-one",
        )
        [
            [
                "id",
                "sales",
            ]
        ]
    )

    if (len(submission) != len(test)):
        raise RuntimeError("Submission row count differs from test.")

    if(submission["sales"].isna().any()):
        raise RuntimeError("Submission contains missing predictions.")

    if(submission["sales"] < 0).any():
        raise RuntimeError("Submission contains negative sales.")

    return submission

def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> GlobalSalesLSTM:
    """
    Load a GlobalSalesLSTM model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint file containing model state and configuration.
        device: Torch device to map the model to.

    Returns:
        GlobalSalesLSTM: The loaded model ready for inference.
"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model = GlobalSalesLSTM(
        **checkpoint["model_config"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    return model

def run_inference(
    checkpoint_path: Path,
    preprocessor_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Run inference using a trained GlobalSalesLSTM model and a preprocessor.

    Args:
        checkpoint_path: Path to the model checkpoint file.
        preprocessor_path: Path to the preprocessor file.
        output_path: Path to save the submission CSV file.

    Returns:
        pd.DataFrame: The submission DataFrame containing predictions.
    """
    config = Config()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ----------------------------
    # Load row data
    # ---------------------------
    data = load_raw_data(config.data_dir)
    validate_raw_data(data)

    split = get_temporal_split(
        data["train"],
        data["test"],
    )

    # -------------------------
    # Features
    # -------------------------

    panel = build_feature_panel(
        data
    )

    preprocessor = (load_preprocessor(preprocessor_path))
    model_frame = (
        transform_panel(
            panel,
            preprocessor,
        )
    )
    validate_model_frame(model_frame)

    # -----------------------
    # Inference dataset
    # -----------------------
    inference_dataset = (
        StoreSalesWindowsDataset(
            df=model_frame,
            context_length=config.context_length,
            horizon=split.horizon,
            hist_features=HIST_FEATURES,
            future_features=FUTURE_FEATURES,
            static_features=STATIC_CAT_FEATURES,
            exact_target_start=split.test_start,
            stride=1,
            require_target=False,
        )
    )

    inference_loader = (
        create_dataloader(
            dataset=inference_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
            num_workers=config.num_workers,
        )
    )

    # -----------------------------
    # Model
    # -----------------------------
    model = (
        load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
        )
    )

    predictions = (
        predict_loader(
            model=model,
            loader=inference_loader,
            device=device,
        )
    )
    prediction_df = (
        predictions_to_frame(
            inference_dataset,
            predictions,
        )
    )

    submission = (
        create_submission(
            test=data["test"],
            prediction_df=prediction_df,
        )
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True,)
    submission.to_csv(
        output_path,
        index=False,
    )

    return submission


def main() -> None:
    config = Config()

    parser = (argparse.ArgumentParser())

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            config.artifact_dir / "models" / "final_lstm.pt"
        ),
    )

    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=(
            config.artifact_dir / "predictions" / "submission.csv"
        ),
    )

    args = parser.parse_args()

    submission = run_inference(
        checkpoint_path=args.checkpoint, 
        preprocessor_path=args.preprocessor,
        output_path=args.output,
    )

    print(
        f"Created submission: "
        f"with {len(submission):,} rows."
    )

if __name__ == "__main__":
    main()
