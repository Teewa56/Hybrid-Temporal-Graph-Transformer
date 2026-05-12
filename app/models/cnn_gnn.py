import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PayloadCNN(nn.Module):
    def __init__(self, input_dim: int = 64, num_filters: int = 128, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(1, num_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters // 2, kernel_size, padding=1)
        self.pool  = nn.MaxPool1d(kernel_size=4, stride=4)
        self.flatten_dim = (num_filters // 2) * 16

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)


class CNNGNNHybrid(nn.Module):
    """
    Two-stage hybrid for payment injection detection.
    Stage 1 — CNN: detects payload-level data tampering.
    Stage 2 — GNN layer: validates fund flow logical consistency
               within the current network state.
    Both signals are fused before final classification.
    """

    def __init__(
        self,
        payload_input_dim: int = 64,
        graph_embedding_dim: int = 64,
        num_filters: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.cnn = PayloadCNN(payload_input_dim, num_filters)
        cnn_out_dim = (num_filters // 2) * 16

        # GNN layer: takes pre-computed graph node embedding for this transaction
        self.gnn_projection = nn.Sequential(
            nn.Linear(graph_embedding_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
        )

        # Fusion classifier
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        payload_features: torch.Tensor,
        graph_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            payload_features: (batch, payload_input_dim)
            graph_embedding: (batch, graph_embedding_dim) — node embedding
                              for this transaction from the graph service
        Returns:
            risk_score: (batch, 1)
        """
        cnn_out = self.cnn(payload_features)
        gnn_out = self.gnn_projection(graph_embedding)
        fused = torch.cat([cnn_out, gnn_out], dim=-1)
        return self.fusion(fused)

    @torch.no_grad()
    def predict(
        self,
        payload_features: np.ndarray,
        graph_embedding: np.ndarray,
    ) -> float:
        self.eval()
        pf = torch.tensor(payload_features, dtype=torch.float32).unsqueeze(0)
        ge = torch.tensor(graph_embedding, dtype=torch.float32).unsqueeze(0)
        return self.forward(pf, ge).item()