from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import random

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn

from store_sales.evaluate import (evaluate_rmsle,)

def seed_everything(
    seed: int,
) -> None:
    """
    Seed all relevant random number generators for reproducibility. This ensures that experiments are deterministic and results can be consistently reproduced across different runs.

    Args:
        seed (int): The seed value to use for all random number generators.
    Raises:
        ValueError: If the seed is not a non-negative integer.
    """
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a non-negative integer.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = (False)
    torch.backends.cudnn.deterministic = (True)
    torch.use_deterministic_algorithms(True, warn_only=True)

def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count the number of trainable parameters in a given PyTorch model.

    Args:
        model (nn.Module): The PyTorch model for which to count trainable parameters.

    Returns:
        int: The total number of trainable parameters in the model.
    """
    return sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

def build_optimizers(
    model: nn.Module, 
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    Build and return the optimizer and learning rate scheduler for training a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model to optimize.
        learning_rate (float): The learning rate for the optimizer.
        weight_decay (float): The weight decay (L2 regularization) for the optimizer.

    Returns:
        tuple: A tuple containing the optimizer and the learning rate scheduler.
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    return (
        optimizer,
        scheduler,
    )

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    criterion,
    device: torch.device,
    grad_clip: float,
) -> float:
    """
    Train the model for one epoch.

    Args:
        model (nn.Module): The PyTorch model to train.
        loader: DataLoader providing training batches.
        optimizer: Optimizer for updating model parameters.
        criterion: Loss function.
        device (torch.device): Device to perform computations on.
        grad_clip (float): Maximum gradient norm for clipping.

    Returns:
        float: Average training loss for the epoch.
    
    Raises:
        ValueError: If the training loader contains no samples.
    """
    model.train()

    total_loss = 0.0
    total_samples = 0

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
        y_log = (y_log.to(device, non_blocking=True,))
        target_loc = (target_loc.to(device, non_blocking=True,).unsqueeze(1))
        target_scale = (target_scale.to(device, non_blocking=True,).unsqueeze(1))

        optimizer.zero_grad(set_to_none=True)

        prediction_z = model(
            x_hist,
            x_future,
            static_cat,
        )

        prediction_log = (
            prediction_z * target_scale + target_loc
        )
        loss = criterion(prediction_log, y_log,)

        nn.utils.clip_grad_norm_(
            model.parameters(),
            grad_clip,
        )
        optimizer.step()

        batch_size = (x_hist.size(0))
        total_loss += (loss.item() * batch_size)
        total_samples += (batch_size)

    if total_samples == 0:
        raise ValueError("Training loader contains no samples.")

    return (total_loss / total_samples)

def _save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int, 
    val_rmsle: float | None = None,
) -> None:
    """
    Save the model checkpoint to the specified path.

    Args:
        model (nn.Module): The PyTorch model to save.
        path (Path): Path to save the checkpoint.
        epoch (int): Current epoch number.
        val_rmsle (float | None): Validation RMSLE score, if available.

    Returns:
        None
    """
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
        "epoch": int(epoch),
    }

    if val_rmsle is not None:
        checkpoint["val_rmsle"] = float(val_rmsle)
    
    torch.save(checkpoint, path)

def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    config, 
    device: torch.device,
    checkpoint_path: Path,
    use_mlflow: bool = True,
    run_name: str | None = None,
) -> tuple[pd.DataFrame, float, int]:
    """
    Train the model using the specified training and validation loaders, 
    applying early stopping based on validation RMSLE. Logs training 
    progress and optionally records metrics to MLflow.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Configuration object containing training hyperparameters.
        device (torch.device): Device to perform computations on.
        checkpoint_path (Path): Path to save model checkpoints.
        use_mlflow (bool): Whether to log metrics to MLflow.
        run_name (str | None): Name of the MLflow run.

    Returns:
        tuple[pd.DataFrame, float, int]: Training history as a DataFrame, 
        best validation RMSLE, and the epoch at which it was achieved.
    """

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True,)

    optimizer, scheduler = (
        build_optimizer(
            model=model,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    )

    criterion = nn.MSELoss()
    best_score = float("inf")
    best_epoch = 0

    epochs_without_improvement = 0
    history: list[dict] = []

    # ===================================
    # MLflow
    # ===================================
    if use_mlflow:
        tracking_dir = (
            config.artifact_dir.parent / "mlruns"
        )
        tracking_dir.mkdir(
            parents=True,
            exist_ok=True,
        )
        mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
        mlflow.set_experiment(config.mlflow_experiment)

        run_context = (
            mlflow.start_run(run_name=run_name)
        )

    else:
        run_context = (nullcontext())

    with run_context:
        if use_mlflow:
            mlflow.log_params(
                {
                    "context_length": config.context_length,
                    "train_stride": config.train_stride,
                    "batch_size": config.batch_size,
                    "hidden_size": config.hidden_size,
                    "num_layers": config.num_layers,
                    "dropout": config.dropout,
                    "learning_rate": config.learning_rate,
                    "weight_decay": config.weight_decay,
                    "seed": config.seed,
                    "trainable_parameters": count_trainable_parameters(model,),
                }
            )
        # ===================================
        # Epoch loop
        # ==================================
        for epoch in range(1, config.max_epoch + 1,):
            train_loss = (
                train_one_epoch(
                    model=model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    grad_clip=config.grad_clip,
                )
            )

            val_rmsle = (
                evaluate_rmsle(
                    model=model,
                    loader=val_loader,
                    device=device,
                )
            )
            scheduler.step(val_rmsle)
            learning_rate = (
                optimizer.param_group[0]["lr"]
            )

            history.append(
                {
                    "epoch":epoch,
                    "train_mse":train_loss,
                    "val_rmsle":val_rmsle,
                    "learning_rate": learning_rate,
                }
            )

            print(
                f"Epoch {epoch: 02d} | "
                f"train_mse="
                f"{train_loss:.6f} | "
                f"val_rmsle="
                f"{val_rmsle:.6f} | "
                f"lr="
                f"{learning_rate:.2e}"
            )

            if use_mlflow:
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "val_rmsle": val_rmsle,
                        "learning_rate":learning_rate,
                    }, step=epoch
                )
            
            # =====================================
            # Improvement
            # ====================================
            if (val_rmsle < best_score - config.min_delta):
                best_score = (val_rmsle)
                best_epoch = (epoch)
                epochs_without_improvement = (0)
                
                _save_checkpoint(
                    model=model,
                    path=checkpoint_path,
                    epoch=epoch,
                    val_rmsle=val_rmsle,
                )

            else:
                epochs_without_improvement += 1

            # =====================================
            # Early stopping
            # =====================================
            if (epochs_without_improvement >= config.patience):
                print("Early stopping.")
                break
        
        # =======================================
        # History artifact
        # =======================================
        history_df = (
            pd.DataFrame(history)
        )
        history_path = (checkpoint_path.parent / "training_history.csv")
        history_df.to_csv(
            history_path,
            index=False,
        )
        if use_mlflow:
            mlflow.log_metric(
                "best_val_rmsle",
                best_score,
            )
            mlflow.log_metric(
                "best_epoch",
                best_epoch,
            )
            mlflow.log_artifact(
                str(checkpoint_path)
            )
            mlflow.log_artifact(
                str(history_path)
            )

    # ===============================
    # Restore best model
    # ===============================
    checkpoint=torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return (
        history_df,
        best_score,
        best_epoch
    )

def train_fixed_epochs(
    model:nn.Module,
    train_loader,
    config,
    device: torch.device,
    epochs: int,
    checkpoint_path: Path,
) -> pd.DataFrame:
    """
    Final retraining after model selection.
    No validation/early stopping is performed because all available historical observations are now used.
    This function performs training for a fixed number of epochs without any early stopping mechanism.
    The model's state is saved at the end of training to the specified checkpoint path.
    Args:
        model (nn.Module): The neural network model to train.
        train_loader: DataLoader for the training dataset.
        config: Configuration object containing hyperparameters.
        device (torch.device): Device to run the training on.
        epochs (int): Number of epochs to train for.
        checkpoint_path (Path): Path to save the model checkpoint.

    Returns:
        pd.DataFrame: DataFrame containing the training history with columns "epoch" and "train_mse".
    
    Raises:
        ValueError: If the number of epochs is not positive.
    """

    if epochs <= 0:
        raise ValueError("Number of epochs must be positive.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, epochs + 1,):
        train_loss = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                grad_clip=config.grad_clip,
            )
        )
        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
            }
        )
        print(
            f"Final training "
            f"epoch {epoch:02d}/{epochs} | "
            f"train_mse: {train_loss:.6f}"
        )
    _save_checkpoint(
        model=model,
        checkpoint_path=checkpoint_path,
        epoch=epochs,
    )

    return pd.DataFrame(history)

