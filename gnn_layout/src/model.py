"""GraphSAGE model for semantic role bbox prior prediction."""

from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import SAGEConv, global_mean_pool

from .orientation import ORIENTATIONS
from .roles import NUM_ROLES


class GNNLayoutPredictor(nn.Module):
    """Predict normalized bbox priors for the main semantic roles."""

    def __init__(
        self,
        in_channels: int,
        hidden: int = 128,
        dropout: float = 0.15,
        target_size_dim: int = 3,
        orientation_dim: int = len(ORIENTATIONS),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.dropout_p = dropout

        self.conv1 = SAGEConv(in_channels, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.conv3 = SAGEConv(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Sequential(
            nn.Linear(hidden + target_size_dim + orientation_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, NUM_ROLES * 4),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        x = self.dropout(torch.relu(self.conv1(x, edge_index)))
        x = self.dropout(torch.relu(self.conv2(x, edge_index)))
        x = self.dropout(torch.relu(self.conv3(x, edge_index)))
        graph_embedding = global_mean_pool(x, batch)

        num_graphs = graph_embedding.size(0)
        target_size = data.target_size.view(num_graphs, -1).to(graph_embedding.device)
        target_orientation = data.target_orientation_onehot.view(num_graphs, -1).to(graph_embedding.device)
        decoded = self.decoder(torch.cat([graph_embedding, target_size, target_orientation], dim=-1))
        return torch.sigmoid(decoded).view(num_graphs, NUM_ROLES, 4)
