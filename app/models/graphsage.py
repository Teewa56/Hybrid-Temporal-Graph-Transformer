import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torch_geometric.nn import SAGEConv, to_hetero
    from torch_geometric.data import HeteroData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class GraphSAGEFraudDetector(nn.Module):
    """
    GraphSAGE (inductive GNN) for social engineering and mule network detection.
    Models the payment ecosystem as a heterogeneous graph.
    Can detect fraudulent accounts even if created hours ago —
    by their connection topology to known fraud clusters.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 128,
        out_channels: int = 64,
        num_layers: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels) if PYG_AVAILABLE else nn.Linear(in_channels, hidden_channels))

        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels) if PYG_AVAILABLE else nn.Linear(hidden_channels, hidden_channels))

        self.convs.append(SAGEConv(hidden_channels, out_channels) if PYG_AVAILABLE else nn.Linear(hidden_channels, out_channels))

        self.classifier = nn.Sequential(
            nn.Linear(out_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node feature matrix (num_nodes, in_channels)
            edge_index: Graph connectivity (2, num_edges)
        Returns:
            node_scores: Per-node fraud probability (num_nodes, 1)
        """
        for i, conv in enumerate(self.convs[:-1]):
            if PYG_AVAILABLE:
                x = conv(x, edge_index)
            else:
                x = conv(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        if PYG_AVAILABLE:
            x = self.convs[-1](x, edge_index)
        else:
            x = self.convs[-1](x)

        return self.classifier(x)

    @torch.no_grad()
    def predict_node(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        target_node_idx: int,
    ) -> float:
        """
        Score a specific node (account) in the graph.
        Returns fraud probability for that node.
        """
        self.eval()
        x = torch.tensor(node_features, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        scores = self.forward(x, ei)
        return scores[target_node_idx].item()