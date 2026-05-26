"""Model evaluation utilities for trading ML models.

This module provides comprehensive evaluation metrics specifically designed
for financial machine learning models, including classification metrics,
regression metrics, and trading-specific performance measures.

Example:
    >>> from imst_quant.models.evaluation import evaluate_classification_model
    >>> import polars as pl
    >>> y_true = pl.Series([1, 0, 1, 1, 0])
    >>> y_pred = pl.Series([1, 0, 1, 0, 1])
    >>> metrics = evaluate_classification_model(y_true, y_pred)
    >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
"""

from typing import Dict, Optional, Tuple

import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


def evaluate_classification_model(
    y_true: pl.Series,
    y_pred: pl.Series,
    y_pred_proba: Optional[pl.Series] = None,
) -> Dict[str, float]:
    """Evaluate classification model with comprehensive metrics.

    Args:
        y_true: True labels (binary 0/1 or -1/1).
        y_pred: Predicted labels.
        y_pred_proba: Predicted probabilities (optional, for AUC).

    Returns:
        Dictionary with accuracy, precision, recall, F1, and optionally AUC.
    """
    y_true_np = y_true.to_numpy()
    y_pred_np = y_pred.to_numpy()

    # Handle -1/1 labels by converting to 0/1
    if np.min(y_true_np) < 0:
        y_true_np = (y_true_np + 1) / 2
    if np.min(y_pred_np) < 0:
        y_pred_np = (y_pred_np + 1) / 2

    metrics = {
        "accuracy": accuracy_score(y_true_np, y_pred_np),
        "precision": precision_score(y_true_np, y_pred_np, zero_division=0),
        "recall": recall_score(y_true_np, y_pred_np, zero_division=0),
        "f1_score": f1_score(y_true_np, y_pred_np, zero_division=0),
    }

    if y_pred_proba is not None:
        try:
            y_proba_np = y_pred_proba.to_numpy()
            metrics["roc_auc"] = roc_auc_score(y_true_np, y_proba_np)
        except ValueError:
            pass

    return metrics


def evaluate_regression_model(
    y_true: pl.Series,
    y_pred: pl.Series,
) -> Dict[str, float]:
    """Evaluate regression model with standard metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MSE, RMSE, MAE, and R² metrics.
    """
    y_true_np = y_true.to_numpy()
    y_pred_np = y_pred.to_numpy()

    mse = mean_squared_error(y_true_np, y_pred_np)

    return {
        "mse": mse,
        "rmse": np.sqrt(mse),
        "mae": mean_absolute_error(y_true_np, y_pred_np),
        "r2": r2_score(y_true_np, y_pred_np),
    }


def directional_accuracy(
    y_true_returns: pl.Series,
    y_pred_returns: pl.Series,
) -> float:
    """Calculate directional accuracy for return predictions.

    Args:
        y_true_returns: True return values.
        y_pred_returns: Predicted return values.

    Returns:
        Directional accuracy (0 to 1).
    """
    true_direction = (y_true_returns > 0).cast(int)
    pred_direction = (y_pred_returns > 0).cast(int)

    return float((true_direction == pred_direction).mean())


def profit_factor_from_predictions(
    y_true_returns: pl.Series,
    y_pred_signals: pl.Series,
) -> float:
    """Calculate profit factor from predicted trading signals.

    Args:
        y_true_returns: Actual returns.
        y_pred_signals: Predicted signals (1 for long, -1 for short, 0 for neutral).

    Returns:
        Profit factor (> 1 is profitable).
    """
    pnl = y_true_returns * y_pred_signals

    winning_trades = pnl.filter(pnl > 0)
    losing_trades = pnl.filter(pnl < 0)

    total_wins = float(winning_trades.sum()) if len(winning_trades) > 0 else 0.0
    total_losses = abs(float(losing_trades.sum())) if len(losing_trades) > 0 else 0.0

    if total_losses == 0:
        return float('inf') if total_wins > 0 else 0.0

    return total_wins / total_losses


def evaluate_trading_model(
    y_true_returns: pl.Series,
    y_pred_signals: pl.Series,
    y_pred_proba: Optional[pl.Series] = None,
) -> Dict[str, float]:
    """Comprehensive evaluation for trading models.

    Args:
        y_true_returns: Actual return values.
        y_pred_signals: Predicted signals (1, -1, or 0).
        y_pred_proba: Prediction probabilities (optional).

    Returns:
        Dictionary with comprehensive metrics.
    """
    y_true_direction = (y_true_returns > 0).cast(int)
    y_pred_direction = (y_pred_signals > 0).cast(int)

    classification_metrics = evaluate_classification_model(
        y_true_direction, y_pred_direction, y_pred_proba
    )

    pf = profit_factor_from_predictions(y_true_returns, y_pred_signals)
    da = directional_accuracy(y_true_returns, y_pred_signals)

    return {
        **classification_metrics,
        "directional_accuracy": da,
        "profit_factor": pf,
    }
