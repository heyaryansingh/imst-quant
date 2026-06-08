"""Model explainability utilities for ML trading models.

This module provides model-agnostic explainability methods for understanding
predictions from LSTM, CNN, Transformer, and LightGBM models without external
dependencies like SHAP or LIME.

Implements permutation importance, feature ablation, partial dependence,
feature interaction analysis, and local linear approximations to explain
both global model behavior and individual predictions.

Functions:
    permutation_importance: Measure feature importance via shuffling
    feature_ablation_study: Measure impact by replacing features with baselines
    partial_dependence: Calculate partial dependence for a single feature
    feature_interaction_strength: Measure interaction between feature pairs
    local_explanation: LIME-like local linear explanation for single predictions
    explain_prediction: Generate comprehensive explainability report
    compare_feature_importance_methods: Compare permutation and ablation rankings

Example:
    >>> import numpy as np
    >>> from imst_quant.utils.model_explainability import explain_prediction
    >>> # Assume model.predict is your prediction function
    >>> report = explain_prediction(
    ...     model_fn=model.predict,
    ...     X=X_test,
    ...     instance_idx=0,
    ...     feature_names=["momentum", "volatility", "volume", "rsi"],
    ...     y=y_test,
    ... )
    >>> print(f"Top drivers: {report.top_drivers}")
    >>> print(f"Confidence: {report.prediction_confidence:.2%}")
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats


@dataclass
class FeatureImportance:
    """Importance score for a single feature.

    Attributes:
        feature_name: Name of the feature.
        importance: Raw importance score (higher = more important).
        direction: Whether feature pushes predictions "positive", "negative", or "mixed".
        rank: Rank among all features (1 = most important).
    """

    feature_name: str
    importance: float
    direction: str
    rank: int


@dataclass
class ExplainabilityReport:
    """Comprehensive model explanation report.

    Attributes:
        method: Explanation method used (e.g., "combined", "permutation").
        feature_importances: List of feature importance scores.
        top_drivers: Names of top driving features.
        prediction_confidence: Confidence score for the prediction (0-1).
        model_type: Type of model being explained.
    """

    method: str
    feature_importances: List[FeatureImportance]
    top_drivers: List[str]
    prediction_confidence: float
    model_type: str = "unknown"


def _default_accuracy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Default accuracy metric for classification tasks.

    Args:
        y_true: True labels.
        y_pred: Predicted labels or probabilities.

    Returns:
        Accuracy score between 0 and 1.
    """
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        # Multi-class: take argmax
        y_pred = np.argmax(y_pred, axis=1)
    elif y_pred.ndim > 1:
        y_pred = y_pred.ravel()

    # Handle probability outputs for binary classification
    if y_pred.dtype == np.float64 or y_pred.dtype == np.float32:
        if np.all((y_pred >= 0) & (y_pred <= 1)):
            y_pred = (y_pred > 0.5).astype(int)

    y_true = y_true.ravel()
    y_pred = y_pred.ravel()

    return float(np.mean(y_true == y_pred))


def _mse_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error metric (negated so higher is better).

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        Negative MSE (so higher = better, consistent with accuracy).
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel() if y_pred.ndim > 1 else y_pred
    return -float(np.mean((y_true - y_pred) ** 2))


def permutation_importance(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 10,
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    random_state: Optional[int] = None,
) -> List[FeatureImportance]:
    """Compute permutation importance by shuffling features.

    For each feature, shuffles its values across samples and measures
    the decrease in model performance. Features causing larger drops
    are more important.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values of shape (n_samples,) or (n_samples, n_classes).
        feature_names: Names for each feature column.
        n_repeats: Number of permutation repeats per feature. Default 10.
        metric_fn: Scoring function(y_true, y_pred) -> float. Higher = better.
            Defaults to accuracy for classification.
        random_state: Random seed for reproducibility.

    Returns:
        List of FeatureImportance sorted by importance (highest first).

    Example:
        >>> importances = permutation_importance(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     y=y_test,
        ...     feature_names=["feature_1", "feature_2", "feature_3"],
        ...     n_repeats=5,
        ... )
        >>> for fi in importances[:3]:
        ...     print(f"{fi.feature_name}: {fi.importance:.4f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    if metric_fn is None:
        metric_fn = _default_accuracy_metric

    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) must match "
            f"number of features ({n_features})"
        )

    # Baseline performance
    baseline_pred = model_fn(X)
    baseline_score = metric_fn(y, baseline_pred)

    importances = []

    for idx, name in enumerate(feature_names):
        scores_drop = []

        for _ in range(n_repeats):
            # Create copy and shuffle the feature
            X_permuted = X.copy()
            X_permuted[:, idx] = np.random.permutation(X_permuted[:, idx])

            # Score with permuted feature
            permuted_pred = model_fn(X_permuted)
            permuted_score = metric_fn(y, permuted_pred)

            # Importance = drop in performance
            scores_drop.append(baseline_score - permuted_score)

        mean_drop = float(np.mean(scores_drop))

        # Determine direction by looking at correlation with predictions
        feature_vals = X[:, idx]
        predictions = baseline_pred.ravel() if baseline_pred.ndim > 1 else baseline_pred
        if len(predictions) == len(feature_vals):
            corr = np.corrcoef(feature_vals, predictions)[0, 1]
            if np.isnan(corr):
                direction = "mixed"
            elif corr > 0.1:
                direction = "positive"
            elif corr < -0.1:
                direction = "negative"
            else:
                direction = "mixed"
        else:
            direction = "mixed"

        importances.append(
            FeatureImportance(
                feature_name=name,
                importance=abs(mean_drop),
                direction=direction,
                rank=0,  # Will be set after sorting
            )
        )

    # Sort by importance and assign ranks
    importances.sort(key=lambda x: x.importance, reverse=True)
    for rank, fi in enumerate(importances, start=1):
        fi.rank = rank

    return importances


def feature_ablation_study(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    baseline: str = "zero",
    metric_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
) -> List[FeatureImportance]:
    """Measure feature importance by replacing features with baselines.

    For each feature, replaces all its values with a baseline (zero, mean,
    or random) and measures the change in model performance.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix of shape (n_samples, n_features).
        y: Target values.
        feature_names: Names for each feature column.
        baseline: Replacement strategy - "zero", "mean", or "random".
        metric_fn: Scoring function(y_true, y_pred) -> float. Higher = better.

    Returns:
        List of FeatureImportance sorted by importance (highest first).

    Example:
        >>> importances = feature_ablation_study(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     y=y_test,
        ...     feature_names=features,
        ...     baseline="mean",
        ... )
    """
    if metric_fn is None:
        metric_fn = _default_accuracy_metric

    X = np.asarray(X)
    y = np.asarray(y)

    n_samples, n_features = X.shape

    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) must match "
            f"number of features ({n_features})"
        )

    valid_baselines = {"zero", "mean", "random"}
    if baseline not in valid_baselines:
        raise ValueError(f"baseline must be one of {valid_baselines}, got '{baseline}'")

    # Baseline performance
    baseline_pred = model_fn(X)
    baseline_score = metric_fn(y, baseline_pred)

    importances = []

    for idx, name in enumerate(feature_names):
        X_ablated = X.copy()

        if baseline == "zero":
            X_ablated[:, idx] = 0.0
        elif baseline == "mean":
            X_ablated[:, idx] = np.mean(X[:, idx])
        elif baseline == "random":
            X_ablated[:, idx] = np.random.choice(X[:, idx], size=n_samples)

        # Score with ablated feature
        ablated_pred = model_fn(X_ablated)
        ablated_score = metric_fn(y, ablated_pred)

        # Importance = drop in performance
        score_drop = baseline_score - ablated_score

        # Determine direction
        feature_vals = X[:, idx]
        predictions = baseline_pred.ravel() if baseline_pred.ndim > 1 else baseline_pred
        if len(predictions) == len(feature_vals):
            corr = np.corrcoef(feature_vals, predictions)[0, 1]
            if np.isnan(corr):
                direction = "mixed"
            elif corr > 0.1:
                direction = "positive"
            elif corr < -0.1:
                direction = "negative"
            else:
                direction = "mixed"
        else:
            direction = "mixed"

        importances.append(
            FeatureImportance(
                feature_name=name,
                importance=abs(score_drop),
                direction=direction,
                rank=0,
            )
        )

    # Sort by importance and assign ranks
    importances.sort(key=lambda x: x.importance, reverse=True)
    for rank, fi in enumerate(importances, start=1):
        fi.rank = rank

    return importances


def partial_dependence(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_idx: int,
    grid_points: int = 50,
    feature_name: Optional[str] = None,
    percentile_range: Tuple[float, float] = (5.0, 95.0),
) -> Dict:
    """Calculate partial dependence for a single feature.

    Shows the marginal effect of a feature on predictions by averaging
    predictions over the distribution of other features.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix of shape (n_samples, n_features).
        feature_idx: Index of the feature to analyze.
        grid_points: Number of grid points for the feature. Default 50.
        feature_name: Optional name for the feature.
        percentile_range: Percentile range for grid. Default (5, 95).

    Returns:
        Dictionary containing:
        - feature_values: Grid of feature values
        - predictions: Average predictions at each grid point
        - feature_name: Name of the feature (if provided)
        - std_predictions: Standard deviation of predictions at each point

    Example:
        >>> pd_result = partial_dependence(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     feature_idx=0,
        ...     grid_points=30,
        ...     feature_name="momentum",
        ... )
        >>> import matplotlib.pyplot as plt
        >>> plt.plot(pd_result["feature_values"], pd_result["predictions"])
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if feature_idx < 0 or feature_idx >= n_features:
        raise ValueError(
            f"feature_idx {feature_idx} out of range [0, {n_features - 1}]"
        )

    # Create grid of feature values
    feature_col = X[:, feature_idx]
    low_pct, high_pct = percentile_range
    feature_min = np.percentile(feature_col, low_pct)
    feature_max = np.percentile(feature_col, high_pct)

    # Handle edge case where min == max
    if feature_min == feature_max:
        feature_min = feature_col.min()
        feature_max = feature_col.max()
        if feature_min == feature_max:
            feature_max = feature_min + 1e-6

    grid_values = np.linspace(feature_min, feature_max, grid_points)

    avg_predictions = []
    std_predictions = []

    for grid_val in grid_values:
        # Create dataset with feature fixed to grid value
        X_modified = X.copy()
        X_modified[:, feature_idx] = grid_val

        # Get predictions and average
        preds = model_fn(X_modified)
        if preds.ndim > 1:
            preds = preds.mean(axis=1) if preds.shape[1] > 1 else preds.ravel()

        avg_predictions.append(float(np.mean(preds)))
        std_predictions.append(float(np.std(preds)))

    result = {
        "feature_values": grid_values.tolist(),
        "predictions": avg_predictions,
        "std_predictions": std_predictions,
    }

    if feature_name is not None:
        result["feature_name"] = feature_name

    return result


def feature_interaction_strength(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    feature_i: int,
    feature_j: int,
    grid_points: int = 20,
) -> float:
    """Measure interaction strength between two features.

    Uses the H-statistic approach: measures how much of the joint effect
    of two features cannot be explained by their individual effects.

    H^2 = sum((PD_ij - PD_i - PD_j)^2) / sum(PD_ij^2)

    Values close to 0 indicate no interaction; values close to 1 indicate
    strong interaction.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix of shape (n_samples, n_features).
        feature_i: Index of first feature.
        feature_j: Index of second feature.
        grid_points: Number of grid points per feature. Default 20.

    Returns:
        H-statistic value between 0 and 1.

    Example:
        >>> h_stat = feature_interaction_strength(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     feature_i=0,
        ...     feature_j=1,
        ... )
        >>> print(f"Interaction strength: {h_stat:.3f}")
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if feature_i < 0 or feature_i >= n_features:
        raise ValueError(f"feature_i {feature_i} out of range")
    if feature_j < 0 or feature_j >= n_features:
        raise ValueError(f"feature_j {feature_j} out of range")
    if feature_i == feature_j:
        raise ValueError("feature_i and feature_j must be different")

    # Create grids for both features
    col_i = X[:, feature_i]
    col_j = X[:, feature_j]

    grid_i = np.linspace(np.percentile(col_i, 5), np.percentile(col_i, 95), grid_points)
    grid_j = np.linspace(np.percentile(col_j, 5), np.percentile(col_j, 95), grid_points)

    # Handle degenerate cases
    if grid_i[0] == grid_i[-1]:
        grid_i = np.linspace(col_i.min(), col_i.max() + 1e-6, grid_points)
    if grid_j[0] == grid_j[-1]:
        grid_j = np.linspace(col_j.min(), col_j.max() + 1e-6, grid_points)

    # Calculate individual partial dependences
    pd_i = {}
    for val in grid_i:
        X_mod = X.copy()
        X_mod[:, feature_i] = val
        preds = model_fn(X_mod)
        if preds.ndim > 1:
            preds = preds.mean(axis=1) if preds.shape[1] > 1 else preds.ravel()
        pd_i[val] = float(np.mean(preds))

    pd_j = {}
    for val in grid_j:
        X_mod = X.copy()
        X_mod[:, feature_j] = val
        preds = model_fn(X_mod)
        if preds.ndim > 1:
            preds = preds.mean(axis=1) if preds.shape[1] > 1 else preds.ravel()
        pd_j[val] = float(np.mean(preds))

    # Calculate joint partial dependence and H-statistic
    numerator = 0.0
    denominator = 0.0

    for val_i in grid_i:
        for val_j in grid_j:
            # Joint PD
            X_mod = X.copy()
            X_mod[:, feature_i] = val_i
            X_mod[:, feature_j] = val_j
            preds = model_fn(X_mod)
            if preds.ndim > 1:
                preds = preds.mean(axis=1) if preds.shape[1] > 1 else preds.ravel()
            pd_ij = float(np.mean(preds))

            # Residual after removing individual effects
            residual = pd_ij - pd_i[val_i] - pd_j[val_j]

            numerator += residual ** 2
            denominator += pd_ij ** 2

    if denominator < 1e-10:
        return 0.0

    h_squared = numerator / denominator
    return float(min(1.0, max(0.0, h_squared)))


def local_explanation(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    instance_idx: int,
    feature_names: List[str],
    n_perturbations: int = 500,
    kernel_width: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Dict:
    """Generate LIME-like local explanation for a single prediction.

    Perturbs the instance, weights perturbations by proximity, and fits
    a local linear model to approximate the model around that point.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix (used for perturbation distribution).
        instance_idx: Index of the instance to explain.
        feature_names: Names for each feature.
        n_perturbations: Number of perturbations to generate. Default 500.
        kernel_width: Width for exponential kernel. Default sqrt(n_features) * 0.75.
        random_state: Random seed for reproducibility.

    Returns:
        Dictionary containing:
        - feature_contributions: Dict[feature_name, coefficient]
        - intercept: Linear model intercept
        - local_r2: R-squared of local model
        - prediction: Original model prediction for the instance

    Example:
        >>> explanation = local_explanation(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     instance_idx=42,
        ...     feature_names=feature_names,
        ... )
        >>> for name, contrib in explanation["feature_contributions"].items():
        ...     print(f"{name}: {contrib:+.4f}")
    """
    if random_state is not None:
        np.random.seed(random_state)

    X = np.asarray(X)
    n_samples, n_features = X.shape

    if instance_idx < 0 or instance_idx >= n_samples:
        raise ValueError(f"instance_idx {instance_idx} out of range [0, {n_samples - 1}]")

    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) must match "
            f"number of features ({n_features})"
        )

    instance = X[instance_idx].copy()

    # Get original prediction
    original_pred = model_fn(instance.reshape(1, -1))
    if original_pred.ndim > 1:
        if original_pred.shape[1] > 1:
            # Multi-class: use probability of predicted class
            pred_class = np.argmax(original_pred[0])
            original_value = float(original_pred[0, pred_class])
        else:
            original_value = float(original_pred[0, 0])
    else:
        original_value = float(original_pred[0])

    # Calculate feature statistics for perturbations
    feature_means = np.mean(X, axis=0)
    feature_stds = np.std(X, axis=0)
    feature_stds = np.where(feature_stds < 1e-10, 1.0, feature_stds)

    # Generate perturbations around the instance
    perturbations = np.zeros((n_perturbations, n_features))
    for i in range(n_features):
        # Sample from distribution centered at instance value
        perturbations[:, i] = np.random.normal(
            instance[i], feature_stds[i] * 0.5, n_perturbations
        )

    # Get predictions for perturbations
    perturbed_preds = model_fn(perturbations)
    if perturbed_preds.ndim > 1:
        if perturbed_preds.shape[1] > 1:
            # Multi-class: use same class as original
            pred_class = np.argmax(model_fn(instance.reshape(1, -1))[0])
            perturbed_values = perturbed_preds[:, pred_class]
        else:
            perturbed_values = perturbed_preds.ravel()
    else:
        perturbed_values = perturbed_preds

    # Calculate distances from original instance (normalized)
    distances = np.sqrt(
        np.sum(((perturbations - instance) / feature_stds) ** 2, axis=1)
    )

    # Exponential kernel weights
    if kernel_width is None:
        kernel_width = np.sqrt(n_features) * 0.75

    weights = np.exp(-(distances ** 2) / (kernel_width ** 2))

    # Fit weighted linear regression
    # Normalize features for regression
    X_local = (perturbations - instance) / feature_stds

    # Add intercept
    X_local_with_intercept = np.column_stack([np.ones(n_perturbations), X_local])

    # Weighted least squares: (X'WX)^-1 X'Wy
    W = np.diag(weights)

    try:
        XtWX = X_local_with_intercept.T @ W @ X_local_with_intercept
        XtWy = X_local_with_intercept.T @ W @ perturbed_values
        coefficients = np.linalg.solve(XtWX, XtWy)
    except np.linalg.LinAlgError:
        # Fallback to least squares
        coefficients = np.linalg.lstsq(
            X_local_with_intercept * np.sqrt(weights).reshape(-1, 1),
            perturbed_values * np.sqrt(weights),
            rcond=None,
        )[0]

    intercept = coefficients[0]
    feature_coeffs = coefficients[1:]

    # Calculate local R-squared (weighted)
    y_pred_local = X_local_with_intercept @ coefficients
    ss_res = np.sum(weights * (perturbed_values - y_pred_local) ** 2)
    y_mean = np.sum(weights * perturbed_values) / np.sum(weights)
    ss_tot = np.sum(weights * (perturbed_values - y_mean) ** 2)
    local_r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
    local_r2 = float(max(0.0, min(1.0, local_r2)))

    # Scale coefficients back to original feature scale
    feature_contributions = {
        name: float(coef / feature_stds[i])
        for i, (name, coef) in enumerate(zip(feature_names, feature_coeffs))
    }

    return {
        "feature_contributions": feature_contributions,
        "intercept": float(intercept),
        "local_r2": local_r2,
        "prediction": original_value,
    }


def explain_prediction(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    instance_idx: int,
    feature_names: List[str],
    y: Optional[np.ndarray] = None,
    n_permutation_repeats: int = 5,
    n_local_perturbations: int = 300,
    top_k: int = 5,
    model_type: str = "unknown",
) -> ExplainabilityReport:
    """Generate comprehensive explainability report for a prediction.

    Combines global importance (permutation) with local explanation to
    provide a complete picture of what drives a specific prediction.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix.
        instance_idx: Index of instance to explain.
        feature_names: Names for each feature.
        y: Target values (optional, improves permutation importance accuracy).
        n_permutation_repeats: Repeats for permutation importance. Default 5.
        n_local_perturbations: Perturbations for local explanation. Default 300.
        top_k: Number of top features to include in top_drivers. Default 5.
        model_type: Type of model for reporting. Default "unknown".

    Returns:
        ExplainabilityReport with combined global and local explanations.

    Example:
        >>> report = explain_prediction(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     instance_idx=0,
        ...     feature_names=features,
        ...     y=y_test,
        ...     model_type="LightGBM",
        ... )
        >>> print(f"Method: {report.method}")
        >>> print(f"Top drivers: {report.top_drivers}")
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape

    if instance_idx < 0 or instance_idx >= n_samples:
        raise ValueError(f"instance_idx {instance_idx} out of range")

    # Get local explanation
    local_exp = local_explanation(
        model_fn=model_fn,
        X=X,
        instance_idx=instance_idx,
        feature_names=feature_names,
        n_perturbations=n_local_perturbations,
    )

    # Get global importance if y is provided
    if y is not None:
        y = np.asarray(y)
        global_importances = permutation_importance(
            model_fn=model_fn,
            X=X,
            y=y,
            feature_names=feature_names,
            n_repeats=n_permutation_repeats,
        )

        # Combine global and local: scale local by global importance
        global_imp_dict = {fi.feature_name: fi.importance for fi in global_importances}
        max_global = max(global_imp_dict.values()) if global_imp_dict else 1.0
        max_global = max_global if max_global > 0 else 1.0

        combined_importance = []
        for name in feature_names:
            local_contrib = abs(local_exp["feature_contributions"].get(name, 0))
            global_imp = global_imp_dict.get(name, 0) / max_global

            # Combined score: weighted average
            combined = 0.6 * local_contrib + 0.4 * global_imp

            # Direction from local contribution
            local_val = local_exp["feature_contributions"].get(name, 0)
            if local_val > 0.01:
                direction = "positive"
            elif local_val < -0.01:
                direction = "negative"
            else:
                # Use global direction
                for fi in global_importances:
                    if fi.feature_name == name:
                        direction = fi.direction
                        break
                else:
                    direction = "mixed"

            combined_importance.append(
                FeatureImportance(
                    feature_name=name,
                    importance=combined,
                    direction=direction,
                    rank=0,
                )
            )
    else:
        # Use only local explanation
        combined_importance = []
        for name in feature_names:
            contrib = local_exp["feature_contributions"].get(name, 0)
            if contrib > 0.01:
                direction = "positive"
            elif contrib < -0.01:
                direction = "negative"
            else:
                direction = "mixed"

            combined_importance.append(
                FeatureImportance(
                    feature_name=name,
                    importance=abs(contrib),
                    direction=direction,
                    rank=0,
                )
            )

    # Sort and assign ranks
    combined_importance.sort(key=lambda x: x.importance, reverse=True)
    for rank, fi in enumerate(combined_importance, start=1):
        fi.rank = rank

    # Top drivers
    top_drivers = [fi.feature_name for fi in combined_importance[:top_k]]

    # Prediction confidence based on local R2 and prediction spread
    confidence = local_exp["local_r2"]

    return ExplainabilityReport(
        method="combined_global_local" if y is not None else "local_linear",
        feature_importances=combined_importance,
        top_drivers=top_drivers,
        prediction_confidence=confidence,
        model_type=model_type,
    )


def compare_feature_importance_methods(
    model_fn: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_permutation_repeats: int = 10,
    ablation_baseline: str = "mean",
) -> Dict:
    """Compare permutation and ablation importance methods.

    Runs both methods and computes rank correlation to assess agreement.
    Identifies consensus top features that both methods rank highly.

    Args:
        model_fn: Function that takes X and returns predictions.
        X: Feature matrix.
        y: Target values.
        feature_names: Names for each feature.
        n_permutation_repeats: Repeats for permutation importance.
        ablation_baseline: Baseline strategy for ablation ("zero", "mean", "random").

    Returns:
        Dictionary containing:
        - permutation: List[FeatureImportance] from permutation method
        - ablation: List[FeatureImportance] from ablation method
        - rank_correlation: Spearman correlation between method rankings
        - consensus_top_5: Features in top 5 for both methods

    Example:
        >>> comparison = compare_feature_importance_methods(
        ...     model_fn=model.predict,
        ...     X=X_test,
        ...     y=y_test,
        ...     feature_names=features,
        ... )
        >>> print(f"Rank correlation: {comparison['rank_correlation']:.3f}")
        >>> print(f"Consensus top 5: {comparison['consensus_top_5']}")
    """
    X = np.asarray(X)
    y = np.asarray(y)

    # Run permutation importance
    perm_importances = permutation_importance(
        model_fn=model_fn,
        X=X,
        y=y,
        feature_names=feature_names,
        n_repeats=n_permutation_repeats,
    )

    # Run ablation study
    ablation_importances = feature_ablation_study(
        model_fn=model_fn,
        X=X,
        y=y,
        feature_names=feature_names,
        baseline=ablation_baseline,
    )

    # Create rank dictionaries
    perm_ranks = {fi.feature_name: fi.rank for fi in perm_importances}
    ablation_ranks = {fi.feature_name: fi.rank for fi in ablation_importances}

    # Compute Spearman rank correlation
    perm_rank_values = [perm_ranks[name] for name in feature_names]
    ablation_rank_values = [ablation_ranks[name] for name in feature_names]

    if len(feature_names) >= 2:
        spearman_corr, _ = stats.spearmanr(perm_rank_values, ablation_rank_values)
        rank_correlation = float(spearman_corr)
    else:
        rank_correlation = 1.0 if len(feature_names) == 1 else float("nan")

    # Find consensus top 5
    perm_top_5 = {fi.feature_name for fi in perm_importances[:5]}
    ablation_top_5 = {fi.feature_name for fi in ablation_importances[:5]}
    consensus_top_5 = list(perm_top_5.intersection(ablation_top_5))

    # Sort consensus by average rank
    if consensus_top_5:
        consensus_top_5.sort(
            key=lambda name: (perm_ranks[name] + ablation_ranks[name]) / 2
        )

    return {
        "permutation": perm_importances,
        "ablation": ablation_importances,
        "rank_correlation": rank_correlation,
        "consensus_top_5": consensus_top_5,
    }
