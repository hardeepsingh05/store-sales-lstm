from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def rmsle_from_sales(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE) between the true and predicted sales values.

    Args:
        y_true (np.ndarray): Array of true sales values.
        y_pred (np.ndarray): Array of predicted sales values.

    Returns:
        float: The computed RMSLE between the true and predicted sales values.
    """

    y_true = np.clip(y_true, 0, None)
    y_pred = np.clip(y_pred, 0, None)

    error = (np.log1p(y_pred) - np.log1p(y_true))

    return float(
        np.sqrt(np.mean(error ** 2))
    )

@torch.inference_mode()
def evaluate_rmsle(
    model, 
    loader,
    device: torch.device,
) -> float:
    """
    Evaluate the model using the Root Mean Squared Logarithmic Error (RMSLE) on the provided data loader.

    Args:
        model: The PyTorch model to evaluate.
        loader: DataLoader providing batches of evaluation data.
        device (torch.device): The device to perform computation on.

    Returns:
        float: The computed RMSLE over the evaluation dataset.

    Raises:
        ValueError: If the evaluation loader contains no observations.
    """
    model.eval()

    squared_error_sum = 0.0
    observation_count = 0

    for batch in loader:
        (
            x_hist,
            x_future,
            static_cat,
            y_log,
            target_loc,
            target_scale,
        ) = batch 
        
        x_hist = (x_hist.to(device, non_blocking=True,))
        x_future = (x_future.to(device, non_blocking=True,))
        static_cat = (static_cat.to(device, non_blocking=True,))
        target_loc = (target_loc.to(device, non_blocking=True,))
        target_scale = (target_scale.to(device, non_blocking=True,).unsqueeze(1))

        prediction_z = model(
            x_hist,
            x_future,
            static_cat,
        )

        prediction_log = (
            prediction_z * target_scale + target_loc
        )

        # log1p(sales) cannot be < 0 
        prediction_log = (torch.clamp(prediction_log, min=0.0))

        squared_error_sum += (
            torch.sum((prediction_log - y_log) ** 2).item()
        )
        observation_count += y_log.numel()

    if observation_count == 0:
        raise ValueError('Validation loader contains no observations.')

    mse = (squared_error_sum / observation_count)

    return float(np.sqrt(mse))

def weekly_naive_score(
    df: pd.DataFrame,
    val_start: pd.Timestamp,
    horizon: int,
) -> float:
    """
    Compute the RMSLE score for a naive weekly baseline, where each future day's prediction is based on the corresponding day from the previous week.
    
    Args:
        df (pd.DataFrame): DataFrame containing historical sales data with columns 'store_nbr', 'family', 'date', and 'sales'.
        val_start (pd.Timestamp): The start date of the validation period.
        horizon (int): The number of future days to predict.

    Returns:
        float: The RMSLE score of the naive weekly predictions compared to the actual sales.

    Raises:
        ValueError: If unable to construct the naive baseline due to insufficient historical data.
    """

    val_start = pd.Timestamp(val_start)

    predictions = []
    actuals = []

    for _,group in df.groupby(
        ["store_nbr", "family"]
    ):
        group = group.sort_values("date")
        historical = (
            group.loc[
                (group["date"] >= val_start - pd.Timestamp(days=7)) & (group["date"] < val_start), "sales",].to_numpy(dtype=np.float64)
            )

        future = (
            group.loc[
                (group["date"] >= val_start) & (group["date"] < val_start + pd.Timedelta(days=horizon)),"sales",].to_numpy(dtype=np.float64)
        )

        if len(historical) != 7:
            continue
        if len(future) != horizon:
            continue
        prediction = np.resize(
            historical,
            horizon,
        )

        predictions.append(prediction)
        actuals.append(future)

    if not predictions:
        raise ValueError("Unable to construct naive baseline")

    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    return rmsle_from_sales(actuals, predictions,)

def build_error_tables(
    prediction_df: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Build error tables summarizing the Root Mean Squared Logarithmic Error (RMSLE) for each grouping.

    Args:
        prediction_df (pd.DataFrame): DataFrame containing predictions and actual sales with columns 'store_nbr', 'family', 'horizon', 'prediction', and 'sales'.

    Returns:
        dict[str, pd.DataFrame]: Dictionary containing RMSLE tables for 'family', 'store', and 'horizon'.

    Raises:
        ValueError: If any of the required columns are missing from the input DataFrame.
    """
    required = {
        "store_nbr",
        "family",
        "horizon",
        "prediction",
        "sales",
    }

    missing = (
        required - set(prediction_df.columns)
    )
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = prediction_df.copy()
    df["sq_log_error"]=(
        np.log1p(df["prediction"].clip(lower=0)) - np.log1p(df["sales"].clip(lower=0))
    ) ** 2

    family = (
        df.groupby("family")["sq_log_error"].mean().pow(0.5).rename("rmsle").sort_values(ascending=False).reset_index()
    )
    store = (
        df.groupby("store_nbr")["sq_log_error"].mean().pow(0.5).rename("rmsle").sort_values(ascending=False).reset_index()
    )
    horizon = (
        df.groupby("horizon")["sq_log_error"].mean().pow(0.5).rename("rmsle").reset_index()
    )

    return {
        "family": family,
        "store": store,
        "horizon": horizon
    }

            
    



    








