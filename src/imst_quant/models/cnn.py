"""CNN classifier for return direction (FORE-02)."""

import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    """1D CNN for sequence classification."""

    def __init__(
        self,
        input_dim: int,
        num_filters: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, num_filters, kernel_size, padding=kernel_size // 2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv(x))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)
