## setting up the configuration for the project
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Config:
    """
    Configuration class
    """
    # Paths
    data_dir: Path = Path("data/raw/store_sales")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    artifact_dir: Path = Path("artifacts")
    # Reproducibility
    seed: int = 42
    # Dataset
    context_length: int = 90
    train_stride: int = 7
    
    batch_size: int = 512
    num_workers: int = 0
    # Model
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float= 0.20
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    max_epochs: int = 30
    patience: int = 5

    grad_clip: float = 1.0

    # Experiment tracking
    mlflow_experiment: str = "store-sales-global-lstm"

