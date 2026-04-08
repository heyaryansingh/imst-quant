"""Transformer-based classifier for return direction prediction (FORE-03).

This module implements a Transformer encoder architecture for sequence
classification, used to predict the direction of asset price movements
based on temporal feature sequences (sentiment scores, market indicators).

The architecture consists of:
    1. Linear projection layer (input_dim -> d_model)
    2. Transformer encoder with multi-head self-attention
    3. Global average pooling over sequence dimension
    4. Dropout and linear classification head

Classes:
    TransformerClassifier: Transformer-based binary/multi-class classifier

Example:
    >>> import torch
    >>> from imst_quant.models.transformer import TransformerClassifier
    >>> model = TransformerClassifier(input_dim=32, num_classes=2)
    >>> x = torch.randn(16, 10, 32)  # batch=16, seq_len=10, features=32
    >>> logits = model(x)  # Shape: (16, 2)
"""

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    """Transformer encoder for sequence classification tasks.

    Uses multi-head self-attention to capture temporal dependencies in
    feature sequences, followed by mean pooling and a classification head.
    Suitable for predicting price direction from time-series features.

    Attributes:
        proj: Linear projection from input_dim to d_model.
        transformer: Stack of TransformerEncoderLayers.
        fc: Final classification linear layer.
        dropout: Dropout layer applied before classification.

    Example:
        >>> model = TransformerClassifier(input_dim=64, d_model=128)
        >>> x = torch.randn(8, 20, 64)  # 8 samples, 20 timesteps, 64 features
        >>> out = model(x)  # Shape: (8, 2)
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        """Initialize the TransformerClassifier.

        Args:
            input_dim: Dimension of input features per timestep.
            d_model: Internal dimension of the transformer. Defaults to 64.
            nhead: Number of attention heads. d_model must be divisible
                by nhead. Defaults to 4.
            num_layers: Number of transformer encoder layers. Defaults to 2.
            dropout: Dropout probability for regularization. Defaults to 0.3.
            num_classes: Number of output classes. Defaults to 2 (binary).
        """
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.fc = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the transformer classifier.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        x = self.proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)
