import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TSSSGCLayer(nn.Module):
    """
    Single Temporal-Spatial-Semantic Graph Convolution layer.
    Combines temporal sequence encoding with spatial-semantic
    node feature aggregation to detect behavioral fingerprint mismatches.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.temporal_gate = nn.GRUCell(in_channels, out_channels)
        self.spatial_transform = nn.Linear(in_channels, out_channels)
        self.semantic_gate = nn.Linear(out_channels * 2, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        h: torch.Tensor,
        neighbor_agg: torch.Tensor,
    ) -> torch.Tensor:
        temporal_out = self.temporal_gate(x, h)
        spatial_out = F.relu(self.spatial_transform(neighbor_agg))
        combined = torch.cat([temporal_out, spatial_out], dim=-1)
        return torch.sigmoid(self.semantic_gate(combined)) * temporal_out


class SIMSwapDetector(nn.Module):
    """
    TSSGC model for detecting SIM Swap Handover Events.

    Monitors device metadata sequences (IMEI, IMSI, carrier, location)
    and detects the semantic mismatch between a new device's fingerprint
    and the historical profile anchored to the same account.

    Key signals:
    - Abrupt IMEI/IMSI change
    - Carrier switch
    - Location radius violation
    - High-value transaction immediately post-swap
    """

    def __init__(
        self,
        device_feature_dim: int = 32,
        account_feature_dim: int = 32,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.device_encoder = nn.Linear(device_feature_dim, hidden_dim)
        self.account_encoder = nn.Linear(account_feature_dim, hidden_dim)

        self.tssgc_layers = nn.ModuleList([
            TSSSGCLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Handover event detector: compares current device vs. account history
        self.mismatch_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        device_sequence: torch.Tensor,
        account_history: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            device_sequence: (batch, seq_len, device_feature_dim)
                             Sequence of device metadata events
            account_history: (batch, account_feature_dim)
                             Long-term account behavioral fingerprint
        Returns:
            sim_swap_score: (batch, 1) — probability of SIM swap event
        """
        batch_size = device_sequence.size(0)
        h = torch.zeros(batch_size, self.hidden_dim, device=device_sequence.device)

        account_enc = F.relu(self.account_encoder(account_history))

        for t in range(device_sequence.size(1)):
            x_t = F.relu(self.device_encoder(device_sequence[:, t, :]))
            for layer in self.tssgc_layers:
                h = layer(x_t, h, account_enc)

        # Final hidden state = representation of the "current device"
        # Compare against account history fingerprint
        combined = torch.cat([h, account_enc], dim=-1)
        return self.mismatch_detector(combined)

    @torch.no_grad()
    def predict(
        self,
        device_sequence: np.ndarray,
        account_history: np.ndarray,
    ) -> float:
        self.eval()
        ds = torch.tensor(device_sequence, dtype=torch.float32).unsqueeze(0)
        ah = torch.tensor(account_history, dtype=torch.float32).unsqueeze(0)
        return self.forward(ds, ah).item()