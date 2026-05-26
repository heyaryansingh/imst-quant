"""Feature importance analysis utilities for trading models.

This module provides methods to analyze and interpret feature importance
in trading ML models, including permutation importance, SHAP-like analysis,
and correlation-based feature ranking.

Example:
    >>> from imst_quant.models.feature_importance import calculate_permutation_importance
    >>> import polars as pl
    >>> X = pl.DataFrame({'feat1': [1, 2, 3], 'feat2': [4, 5, 6]})
    >>> y = pl.Series([0, 1, 1])
    >>> importance = calculate_permutation_importance(X, y, model)
"""

from typing import Callable, Dict, List

import numpy as np
import polars as pl
from sklearn.metrics import accuracy_score


def calculate_permutation_importance(
    X: pl.DataFrame,
    y: pl.Series,
    model: Callable,
    metric: Callable = accuracy_score,
    n_repeats: int = 10,
    random_state: int = 42,
) -> Dict[str, float]:
    """Calculate permutation importance for each feature.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        model: Trained model with predict method.
        metric: Scoring metric (default: accuracy_score).
        n_repeats: Number of permutation repeats (default: 10).
        random_state: Random seed.

    Returns:
        Dictionary mapping feature names to importance scores.
    """
    np.random.seed(random_state)

    # Baseline score
    X_np = X.to_numpy()
    y_np = y.to_numpy()
    y_pred = model.predict(X_np)
    baseline_score = metric(y_np, y_pred)

    importances = {}

    for col_idx, col_name in enumerate(X.columns):
        scores = []

        for _ in range(n_repeats):
            # Permute this feature
            X_permuted = X_np.copy()
            X_permuted[:, col_idx] = np.random.permutation(X_permuted[:, col_idx])

            # Predict with permuted feature
            y_pred_permuted = model.predict(X_permuted)
            permuted_score = metric(y_np, y_pred_permuted)

            # Importance = drop in score
            scores.append(baseline_score - permuted_score)

        importances[col_name] = float(np.mean(scores))

    return importances


def rank_features_by_importance(
    importances: Dict[str, float],
    top_n: int = 10,
) -> pl.DataFrame:
    """Rank features by importance scores.

    Args:
        importances: Dictionary of feature importances.
        top_n: Number of top features to return (default: 10).

    Returns:
        DataFrame with ranked features.
    """
    df = pl.DataFrame({
        "feature": list(importances.keys()),
        "importance": list(importances.values()),
    })

    df = df.sort("importance", descending=True).head(top_n)

    # Add relative importance
    total_importance = df["importance"].sum()
    df = df.with_columns(
        (pl.col("importance") / total_importance).alias("relative_importance")
    )

    return df


def calculate_feature_correlations(
    X: pl.DataFrame,
    y: pl.Series,
) -> Dict[str, float]:
    """Calculate correlation between each feature and target.

    Args:
        X: Feature DataFrame.
        y: Target Series.

    Returns:
        Dictionary mapping feature names to correlation coefficients.
    """
    correlations = {}

    df = X.with_columns(pl.lit(y).alias("target"))

    for col in X.columns:
        corr = df.select(
            pl.corr(col, "target").alias("corr")
        )["corr"][0]

        correlations[col] = float(corr) if corr is not None else 0.0

    return correlations


def identify_redundant_features(
    X: pl.DataFrame,
    threshold: float = 0.95,
) -> List[str]:
    """Identify highly correlated (redundant) features.

    Args:
        X: Feature DataFrame.
        threshold: Correlation threshold for redundancy (default: 0.95).

    Returns:
        List of redundant feature names to potentially drop.
    """
    corr_matrix = X.to_pandas().corr().abs()

    redundant = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > threshold:
                colname = corr_matrix.columns[j]
                if colname not in redundant:
                    redundant.append(colname)

    return redundant


def select_top_features(
    importances: Dict[str, float],
    n_features: int,
) -> List[str]:
    """Select top N most important features.

    Args:
        importances: Dictionary of feature importances.
        n_features: Number of features to select.

    Returns:
        List of selected feature names.
    """
    sorted_features = sorted(
        importances.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return [feat for feat, _ in sorted_features[:n_features]]


def calculate_feature_stability(
    importance_runs: List[Dict[str, float]],
) -> Dict[str, float]:
    """Calculate stability of feature importance across multiple runs.

    Args:
        importance_runs: List of importance dictionaries from different runs.

    Returns:
        Dictionary with mean importance and coefficient of variation.
    """
    if not importance_runs:
        return {}

    # Collect all features
    all_features = set()
    for run in importance_runs:
        all_features.update(run.keys())

    stability = {}

    for feat in all_features:
        values = [run.get(feat, 0.0) for run in importance_runs]

        mean_val = np.mean(values)
        std_val = np.std(values)
        cv = std_val / mean_val if mean_val != 0 else 0.0

        stability[feat] = {
            "mean_importance": mean_val,
            "std_importance": std_val,
            "cv": cv,  # Lower CV = more stable
        }

    return stability
