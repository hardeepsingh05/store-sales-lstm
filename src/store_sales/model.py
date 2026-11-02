from __future__ import annotations

import math
import torch
from torch import nn

def get_embedding_dim(
    cardinality: int,
    ) -> int:
    """
    Small heuristic for choosing embedding dimensions.
    
    Args:
        cardinality (int): The number of unique categories for the categorical feature.
    
    Returns:
        int: The recommended embedding dimension for the given cardinality.
    """

    return min(
        32, 
        max(4, int(math.ceil(cardinality ** 0.25 * 4)),),
    )

def get_embedding_dims(
    cardinalities: list[int],
    ) -> list[int]:
    """
    Get embedding dimensions for a list of categorical feature cardinalities.
    
    Args:
        cardinalities (list[int]): A list containing the number of unique categories for each categorical feature.
    
    Returns:
        list[int]: A list of recommended embedding dimensions for each categorical feature.
    """
    return [
        get_embedding_dim(cardinality) for cardinality in cardinalities
    ]

class GlobalSalesLSTM(nn.Module):
    """
    Global LSTM model.

    Historical sequence -> static embeddings -> LSTM encoder -> final hidden state -> combine with future-known features -> multi-horizon forecast head
    """

    def __init__(
        self,
        hist_dim: int,
        future_dim: int, 
        categorical_cardinalities: list[int], 
        embedding_dims: list[int],
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        """
        Initialize the GlobalSalesLSTM model.

        Args:
            hist_dim (int): Dimension of the historical sequence input.
            future_dim (int): Dimension of the future-known features input.
            categorical_cardinalities (list[int]): List of cardinalities for each categorical feature.
            embedding_dims (list[int]): List of embedding dimensions for each categorical feature.
            hidden_size (int, optional): Hidden size of the LSTM and linear layers. Defaults to 128.
            num_layers (int, optional): Number of LSTM layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__()

        if(len(categorical_cardinalities) != len(embedding_dims)):
            raise ValueError(
                "Cardinalities and embedding_dims must have equal length."
            )

        self.hist_dim = hist_dim
        self.future_dim = future_dim

        self.categorical_cardinalities = (
            categorical_cardinalities
        )

        self.embedding_dims = (embedding_dims)
        self.hidden_size = (hidden_size)
        self.num_layers = (num_layers)
        self.dropout = (dropout)

        # ---------------------------------------
        # Embeddings
        # ---------------------------------------
        self.embeddings = (
            nn.ModuleList(
                [
                    nn.Embedding(num_embeddings=cardinality, embedding_dim=embedding_dim, padding_idx=0,) for (cardinality, embedding_dim,) in zip (categorical_cardinalities, embedding_dims,)
                ]
            )
        )

        total_embedding_dims = sum(
            embedding_dims
        )
        self.total_embedding_dims = (total_embedding_dims)

        # --------------------------------------
        # Input projection
        # ---------------------------------------
        self.hist_projection = (
            nn.Sequential(nn.Linear(
                hist_dim + total_embedding_dims, hidden_size,
                ),
                nn.ReLU(),
            )
        )

        # ---------------------------------------
        # LSTM encoder
        # --------------------------------------
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout = (
                dropout if num_layers > 1 else 0.0
            ),
        )

        # -------------------------------------
        # Forecast head
        # ------------------------------------
        decoder_input_dim = (
            hidden_size + future_dim + total_embedding_dims
        )

        self.forecast_head = (
            nn.Sequential(
                nn.Linear(
                    decoder_input_dim,
                    hidden_size,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(
                    hidden_size,
                    hidden_size //2,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_size // 2,
                    1,
                ),
            )
        )

    def embed_static(
        self,
        static_cat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embed the static categorical features.

        Args:
            static_cat (torch.Tensor): Tensor containing the static categorical features.

        Returns:
            torch.Tensor: Concatenated embeddings for the static categorical features.
        """
        embeddings = []

        for i, embedding_layer in enumerate(
            self.embeddings
        ):
            embedding = (embedding_layer(static_cat[:,i]))
            embeddings.append(embedding)

        return torch.cat(embeddings, dim=-1)

    def forward(
        self,
        x_hist: torch.Tensor,
        x_future: torch.Tensor,
        static_cat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the GlobalSalesLSTM model.

        Args:
            x_hist (torch.Tensor): Historical sequence input.
            x_future (torch.Tensor): Future-known features input.
            static_cat (torch.Tensor): Static categorical features input.

        Returns:
            torch.Tensor: Predicted future values.
        """
        # ---------------------------------
        # Static embeddings
        # ----------------------------------
        static_emb=(
            self.embed_static(static_cat)
        )
        
        # -----------------------------------
        # Historical encoder
        # -----------------------------------
        sequence_length = (
            x_hist.size(1)
        )
        hist_static = (
            staitc_emb.unsqueeze(1).expand(
                -1,
                sequence_length,
                -1
            )
        )

        encoder_input = (torch.cat(
                [
                    x_hist,
                    hist_static,
                ],
                dim = -1,
            )
        )

        encoder_input = (
            self.hist_projection(
                encoder_input
            )
        )

        _, (hidden_state, _,) = self.lstm(encoder_input)

        # hidden_state:
        # [num_layers, batch, hidden_size]

        context = (
            hidden_state[-1]
        )

        # context: [batch, hidden_size]

        # ----------------------------------
        # Multi-horizon direct decoder
        # --------------------------------
        horizon = (
            x_future.size(1)
        )

        context_expanded = (
            context.unsqueeze(1).expand(-1, horizon, -1)
        )

        static_expanded = (
            static_emb.unsqueeze(1).expand(-1,horizon,-1)
        )

        decoder_input = (
            torch.cat(
                [
                    context_expanded,
                    x_future,
                    static_expanded
                ],
                dim = -1,
            )
        )

        prediction_z = (
            self.forcast_head(
                decoder_input
            ).squeeze(-1)
        )

        return prediction_z

    def get_config(
        self,
    ) -> dict:
        """
        Everything necessary to rebuild the architecture.

        Args:
            None
        Returns:
            dict: A dictionary containing all the necessary configuration parameters to rebuild the model.
        """

        return {
            "hist_dim": self.hist_dim,
            "future_dim": self.future_dim,
            "categorical_cardinalities": self.categorical_cardinalities,
            "embedding_dims": self.embedding_dims,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
        }