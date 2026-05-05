"""Batch data preprocessing pipelines for efficient feature engineering.

This module provides utilities to apply multiple preprocessing steps in a single
pass through the data, optimizing performance for large datasets.

Functions:
    create_preprocessing_pipeline: Build a composable preprocessing pipeline
    apply_pipeline: Execute pipeline on data
    get_default_pipeline: Get a pre-configured standard pipeline
    save_pipeline_config: Save pipeline configuration for reproducibility
    load_pipeline_config: Load saved pipeline configuration

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.batch_pipeline import create_preprocessing_pipeline
    >>> df = pl.DataFrame({"close": [100, 102, 98, 105]})
    >>> pipeline = create_preprocessing_pipeline([
    ...     {"type": "normalize", "method": "z-score"},
    ...     {"type": "lag", "columns": ["close"], "lags": [1, 2]},
    ... ])
    >>> result = apply_pipeline(df, pipeline)
"""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import polars as pl
import structlog

logger = structlog.get_logger()


class PreprocessingStep:
    """Base class for preprocessing steps in the pipeline."""

    def __init__(self, name: str, params: Optional[Dict[str, Any]] = None):
        """Initialize preprocessing step.

        Args:
            name: Name of the preprocessing step.
            params: Parameters for the step.
        """
        self.name = name
        self.params = params or {}

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        """Apply the preprocessing step to data.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        raise NotImplementedError("Subclasses must implement apply()")

    def to_dict(self) -> Dict[str, Any]:
        """Convert step to dictionary for serialization."""
        return {"name": self.name, "params": self.params}


class NormalizeStep(PreprocessingStep):
    """Normalize numeric columns using z-score or min-max."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        method = self.params.get("method", "z-score")
        columns = self.params.get("columns", [col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64, pl.Float32, pl.Int32]])

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping normalization")
                continue

            if method == "z-score":
                mean = df[col].mean()
                std = df[col].std()
                if std and std != 0:
                    df = df.with_columns(((pl.col(col) - mean) / std).alias(f"{col}_norm"))
            elif method == "min-max":
                min_val = df[col].min()
                max_val = df[col].max()
                if min_val is not None and max_val is not None and max_val != min_val:
                    df = df.with_columns(
                        ((pl.col(col) - min_val) / (max_val - min_val)).alias(f"{col}_norm")
                    )

        return df


class LagStep(PreprocessingStep):
    """Add lagged features for time series modeling."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        columns = self.params.get("columns", [])
        lags = self.params.get("lags", [1])

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping lag features")
                continue

            for lag in lags:
                df = df.with_columns(pl.col(col).shift(lag).alias(f"{col}_lag{lag}"))

        return df


class RollingStep(PreprocessingStep):
    """Compute rolling statistics for specified columns."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        columns = self.params.get("columns", [])
        windows = self.params.get("windows", [5, 10])
        stats = self.params.get("stats", ["mean", "std"])

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping rolling stats")
                continue

            for window in windows:
                for stat in stats:
                    col_name = f"{col}_roll{window}_{stat}"
                    if stat == "mean":
                        df = df.with_columns(pl.col(col).rolling_mean(window_size=window).alias(col_name))
                    elif stat == "std":
                        df = df.with_columns(pl.col(col).rolling_std(window_size=window).alias(col_name))
                    elif stat == "min":
                        df = df.with_columns(pl.col(col).rolling_min(window_size=window).alias(col_name))
                    elif stat == "max":
                        df = df.with_columns(pl.col(col).rolling_max(window_size=window).alias(col_name))

        return df


class DifferenceStep(PreprocessingStep):
    """Calculate differences or returns for columns."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        columns = self.params.get("columns", [])
        periods = self.params.get("periods", [1])
        log_returns = self.params.get("log_returns", False)

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping differencing")
                continue

            for period in periods:
                if log_returns:
                    # Log returns: ln(price_t / price_t-1)
                    df = df.with_columns(
                        (pl.col(col) / pl.col(col).shift(period)).log().alias(f"{col}_logret{period}")
                    )
                else:
                    # Simple returns: (price_t - price_t-1) / price_t-1
                    df = df.with_columns(
                        ((pl.col(col) - pl.col(col).shift(period)) / pl.col(col).shift(period)).alias(
                            f"{col}_ret{period}"
                        )
                    )

        return df


class OutlierStep(PreprocessingStep):
    """Remove or clip outliers using IQR or z-score method."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        columns = self.params.get("columns", [])
        method = self.params.get("method", "iqr")
        threshold = self.params.get("threshold", 3.0)
        action = self.params.get("action", "remove")  # "remove" or "clip"

        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found, skipping outlier handling")
                continue

            if method == "iqr":
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - threshold * iqr
                upper_bound = q3 + threshold * iqr
            elif method == "z-score":
                mean = df[col].mean()
                std = df[col].std()
                if not std or std == 0:
                    continue
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
            else:
                continue

            if action == "remove":
                df = df.filter((pl.col(col) >= lower_bound) & (pl.col(col) <= upper_bound))
            elif action == "clip":
                df = df.with_columns(
                    pl.col(col).clip(lower_bound, upper_bound).alias(col)
                )

        return df


class FillNaStep(PreprocessingStep):
    """Handle missing values with forward fill, backward fill, or interpolation."""

    def apply(self, df: pl.DataFrame) -> pl.DataFrame:
        method = self.params.get("method", "forward")
        columns = self.params.get("columns", None)  # None = all columns

        if method == "forward":
            if columns:
                for col in columns:
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).forward_fill().alias(col))
            else:
                df = df.fill_nan(None).fill_null(strategy="forward")

        elif method == "backward":
            if columns:
                for col in columns:
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).backward_fill().alias(col))
            else:
                df = df.fill_nan(None).fill_null(strategy="backward")

        elif method == "interpolate":
            if columns:
                for col in columns:
                    if col in df.columns:
                        df = df.with_columns(pl.col(col).interpolate().alias(col))
            else:
                df = df.interpolate()

        elif method == "drop":
            df = df.drop_nulls()

        return df


# Step registry for easy lookup
STEP_REGISTRY: Dict[str, type] = {
    "normalize": NormalizeStep,
    "lag": LagStep,
    "rolling": RollingStep,
    "difference": DifferenceStep,
    "outlier": OutlierStep,
    "fillna": FillNaStep,
}


def create_preprocessing_pipeline(steps: List[Dict[str, Any]]) -> List[PreprocessingStep]:
    """Create a preprocessing pipeline from configuration.

    Args:
        steps: List of step configurations, each with "type" and optional "params".

    Returns:
        List of PreprocessingStep instances ready to apply.

    Example:
        >>> pipeline = create_preprocessing_pipeline([
        ...     {"type": "fillna", "params": {"method": "forward"}},
        ...     {"type": "normalize", "params": {"method": "z-score", "columns": ["close"]}},
        ...     {"type": "lag", "params": {"columns": ["close"], "lags": [1, 5]}},
        ... ])
    """
    pipeline = []

    for step_config in steps:
        step_type = step_config.get("type")
        params = step_config.get("params", {})

        if step_type not in STEP_REGISTRY:
            logger.warning(f"Unknown step type: {step_type}, skipping")
            continue

        step_class = STEP_REGISTRY[step_type]
        pipeline.append(step_class(name=step_type, params=params))

    logger.info(f"Created pipeline with {len(pipeline)} steps")
    return pipeline


def apply_pipeline(
    df: pl.DataFrame,
    pipeline: List[PreprocessingStep],
    verbose: bool = False,
) -> pl.DataFrame:
    """Apply preprocessing pipeline to data.

    Args:
        df: Input DataFrame.
        pipeline: List of PreprocessingStep instances.
        verbose: Whether to log each step.

    Returns:
        Transformed DataFrame.

    Example:
        >>> df = pl.DataFrame({"close": [100, 102, 98, 105]})
        >>> pipeline = create_preprocessing_pipeline([...])
        >>> result = apply_pipeline(df, pipeline, verbose=True)
    """
    result = df

    for i, step in enumerate(pipeline):
        if verbose:
            logger.info(f"Applying step {i+1}/{len(pipeline)}: {step.name}")

        result = step.apply(result)

        if verbose:
            logger.info(f"  Shape after: {result.shape}")

    logger.info(f"Pipeline complete. Final shape: {result.shape}")
    return result


def get_default_pipeline(
    price_cols: Optional[List[str]] = None,
    include_returns: bool = True,
    include_lags: bool = True,
    include_rolling: bool = True,
) -> List[PreprocessingStep]:
    """Get a pre-configured standard preprocessing pipeline.

    Args:
        price_cols: List of price columns (default: ["close"]).
        include_returns: Whether to compute returns.
        include_lags: Whether to add lagged features.
        include_rolling: Whether to add rolling statistics.

    Returns:
        Pre-configured pipeline for time series data.

    Example:
        >>> pipeline = get_default_pipeline(price_cols=["close", "open"])
        >>> df_processed = apply_pipeline(df, pipeline)
    """
    if price_cols is None:
        price_cols = ["close"]

    steps = [
        {"type": "fillna", "params": {"method": "forward"}},
    ]

    if include_returns:
        steps.append(
            {"type": "difference", "params": {"columns": price_cols, "periods": [1], "log_returns": True}}
        )

    if include_lags:
        steps.append(
            {"type": "lag", "params": {"columns": price_cols, "lags": [1, 2, 5, 10]}}
        )

    if include_rolling:
        steps.append(
            {
                "type": "rolling",
                "params": {
                    "columns": price_cols,
                    "windows": [5, 10, 20],
                    "stats": ["mean", "std"],
                },
            }
        )

    steps.append(
        {"type": "normalize", "params": {"method": "z-score", "columns": price_cols}}
    )

    return create_preprocessing_pipeline(steps)


def save_pipeline_config(pipeline: List[PreprocessingStep], filepath: Union[str, Path]) -> None:
    """Save pipeline configuration to JSON for reproducibility.

    Args:
        pipeline: List of PreprocessingStep instances.
        filepath: Path to save configuration file.

    Example:
        >>> pipeline = get_default_pipeline()
        >>> save_pipeline_config(pipeline, "config/preprocessing_pipeline.json")
    """
    config = [step.to_dict() for step in pipeline]

    with open(filepath, "w") as f:
        json.dump({"steps": config}, f, indent=2)

    logger.info(f"Saved pipeline config to {filepath}")


def load_pipeline_config(filepath: Union[str, Path]) -> List[PreprocessingStep]:
    """Load pipeline configuration from JSON.

    Args:
        filepath: Path to configuration file.

    Returns:
        Reconstructed preprocessing pipeline.

    Example:
        >>> pipeline = load_pipeline_config("config/preprocessing_pipeline.json")
        >>> df_processed = apply_pipeline(df, pipeline)
    """
    with open(filepath, "r") as f:
        config = json.load(f)

    steps = config.get("steps", [])
    pipeline = create_preprocessing_pipeline(steps)

    logger.info(f"Loaded pipeline with {len(pipeline)} steps from {filepath}")
    return pipeline
