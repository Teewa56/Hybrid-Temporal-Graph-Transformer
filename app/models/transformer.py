import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    input_dim: int = 32        # Feature dimension per transaction
    d_model: int = 128         # Transformer embedding dim
    nhead: int = 4             # Attention heads
    num_layers: int = 3        # Encoder layers
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 50      # Rolling window size


class BehavioralTransformer(nn.Module):
    """
    Transformer Encoder for behavioral sequence analysis.
    Ingests last 50 transactions and outputs a fraud risk score
    based on deviation from the user's learned spending rhythm.
    """

    def __init__(self, config: TransformerConfig = TransformerConfig()):
        super().__init__()
        self.config = config

        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.positional_encoding = nn.Embedding(config.max_seq_len, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.attention_weights: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) — transaction sequence
        Returns:
            risk_score: (batch, 1) — fraud probability
            attn_weights: attention map for explainability
        """
        batch_size, seq_len, _ = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)

        x = self.input_projection(x) + self.positional_encoding(positions)

        # Register hook to capture attention weights for SHAP
        encoded = self.encoder(x)

        # Use CLS-style pooling: take the last position as the sequence summary
        pooled = encoded[:, -1, :]
        risk_score = self.classifier(pooled)

        return risk_score, encoded

    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> float:
        """
        Single-sample inference. sequence shape: (seq_len, input_dim)
        Returns fraud probability as a float.
        """
        self.eval()
        x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
        score, _ = self.forward(x)
        return score.item()