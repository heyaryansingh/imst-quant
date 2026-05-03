"""Feature normalization and scaling utilities for ML pipelines.

Provides robust scaling methods for financial time series features,
including min-max, z-score, and robust scaling resistant to outliers.

Example:
    >>> import polars as pl
    >>> from imst_quant.utils.feature_scaling import StandardScaler
    >>> df = pl.DataFrame({"feature": [1.0, 2.0, 3.0, 100.0]})
    >>> scaler = StandardScaler()
    >>> df_scaled = scaler.fit_transform(df, ["feature"])
"""

import polars as pl
from typing import List, Optional, Dict, Tuple
import json
from pathlib import Path


class StandardScaler:
    """Z-score normalization (mean=0, std=1).

    Attributes:
        mean_: Dictionary of mean values per column.
        std_: Dictionary of std dev values per column.

    Example:
        >>> scaler = StandardScaler()
        >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
        >>> df_scaled = scaler.fit_transform(df, ["a", "b"])
        >>> df_new = pl.DataFrame({"a": [2.5], "b": [25.0]})
        >>> df_new_scaled = scaler.transform(df_new, ["a", "b"])
    """

    def __init__(self):
        self.mean_: Dict[str, float] = {}
        self.std_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame, columns: List[str]) -> "StandardScaler":
        """Compute mean and std from training data.

        Args:
            df: Training DataFrame.
            columns: Columns to fit scaling parameters on.

        Returns:
            Self for method chaining.
        """
        for col in columns:
            self.mean_[col] = df[col].mean()
            self.std_[col] = df[col].std()

        return self

    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Apply z-score normalization using fitted parameters.

        Args:
            df: DataFrame to transform.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame with scaled columns.
        """
        result = df.clone()

        for col in columns:
            if col not in self.mean_ or col not in self.std_:
                raise ValueError(f"Column '{col}' not fitted. Call fit() first.")

            result = result.with_columns(
                ((pl.col(col) - self.mean_[col]) / self.std_[col]).alias(col)
            )

        return result

    def fit_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Fit and transform in one step.

        Args:
            df: DataFrame to fit and transform.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame.
        """
        self.fit(df, columns)
        return self.transform(df, columns)

    def inverse_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Reverse the scaling transformation.

        Args:
            df: Scaled DataFrame.
            columns: Columns to reverse-scale.

        Returns:
            DataFrame with original scale.
        """
        result = df.clone()

        for col in columns:
            if col not in self.mean_ or col not in self.std_:
                raise ValueError(f"Column '{col}' not fitted.")

            result = result.with_columns(
                (pl.col(col) * self.std_[col] + self.mean_[col]).alias(col)
            )

        return result

    def save(self, path: Path) -> None:
        """Save scaler parameters to JSON file."""
        params = {"mean": self.mean_, "std": self.std_}
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> "StandardScaler":
        """Load scaler parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.mean_ = params["mean"]
        self.std_ = params["std"]
        return self


class MinMaxScaler:
    """Min-max normalization to [0, 1] range.

    Attributes:
        min_: Dictionary of min values per column.
        max_: Dictionary of max values per column.

    Example:
        >>> scaler = MinMaxScaler()
        >>> df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
        >>> df_scaled = scaler.fit_transform(df, ["a"])
        >>> print(df_scaled["a"])  # [0.0, 0.333, 0.666, 1.0]
    """

    def __init__(self):
        self.min_: Dict[str, float] = {}
        self.max_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame, columns: List[str]) -> "MinMaxScaler":
        """Compute min and max from training data.

        Args:
            df: Training DataFrame.
            columns: Columns to fit on.

        Returns:
            Self for chaining.
        """
        for col in columns:
            self.min_[col] = df[col].min()
            self.max_[col] = df[col].max()

        return self

    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Scale to [0, 1] range using fitted min/max.

        Args:
            df: DataFrame to transform.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame.
        """
        result = df.clone()

        for col in columns:
            if col not in self.min_ or col not in self.max_:
                raise ValueError(f"Column '{col}' not fitted. Call fit() first.")

            denom = self.max_[col] - self.min_[col]
            if denom == 0:
                result = result.with_columns(pl.lit(0.0).alias(col))
            else:
                result = result.with_columns(
                    ((pl.col(col) - self.min_[col]) / denom).alias(col)
                )

        return result

    def fit_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df, columns)

    def inverse_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Reverse the min-max scaling."""
        result = df.clone()

        for col in columns:
            if col not in self.min_ or col not in self.max_:
                raise ValueError(f"Column '{col}' not fitted.")

            denom = self.max_[col] - self.min_[col]
            result = result.with_columns(
                (pl.col(col) * denom + self.min_[col]).alias(col)
            )

        return result

    def save(self, path: Path) -> None:
        """Save scaler parameters to JSON file."""
        params = {"min": self.min_, "max": self.max_}
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> "MinMaxScaler":
        """Load scaler parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.min_ = params["min"]
        self.max_ = params["max"]
        return self


class RobustScaler:
    """Robust scaling using median and IQR (resistant to outliers).

    Uses median for centering and interquartile range (IQR) for scaling,
    making it robust to outliers in financial data.

    Attributes:
        median_: Dictionary of median values per column.
        iqr_: Dictionary of IQR values per column.

    Example:
        >>> scaler = RobustScaler()
        >>> df = pl.DataFrame({"price": [100, 102, 105, 1000]})  # 1000 is outlier
        >>> df_scaled = scaler.fit_transform(df, ["price"])
    """

    def __init__(self):
        self.median_: Dict[str, float] = {}
        self.iqr_: Dict[str, float] = {}

    def fit(self, df: pl.DataFrame, columns: List[str]) -> "RobustScaler":
        """Compute median and IQR from training data.

        Args:
            df: Training DataFrame.
            columns: Columns to fit on.

        Returns:
            Self for chaining.
        """
        for col in columns:
            self.median_[col] = df[col].median()
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            self.iqr_[col] = q3 - q1

        return self

    def transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Scale using median and IQR.

        Args:
            df: DataFrame to transform.
            columns: Columns to scale.

        Returns:
            Transformed DataFrame.
        """
        result = df.clone()

        for col in columns:
            if col not in self.median_ or col not in self.iqr_:
                raise ValueError(f"Column '{col}' not fitted. Call fit() first.")

            if self.iqr_[col] == 0:
                result = result.with_columns(pl.lit(0.0).alias(col))
            else:
                result = result.with_columns(
                    ((pl.col(col) - self.median_[col]) / self.iqr_[col]).alias(col)
                )

        return result

    def fit_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Fit and transform in one step."""
        self.fit(df, columns)
        return self.transform(df, columns)

    def inverse_transform(self, df: pl.DataFrame, columns: List[str]) -> pl.DataFrame:
        """Reverse the robust scaling."""
        result = df.clone()

        for col in columns:
            if col not in self.median_ or col not in self.iqr_:
                raise ValueError(f"Column '{col}' not fitted.")

            result = result.with_columns(
                (pl.col(col) * self.iqr_[col] + self.median_[col]).alias(col)
            )

        return result

    def save(self, path: Path) -> None:
        """Save scaler parameters to JSON file."""
        params = {"median": self.median_, "iqr": self.iqr_}
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    def load(self, path: Path) -> "RobustScaler":
        """Load scaler parameters from JSON file."""
        with open(path, "r") as f:
            params = json.load(f)
        self.median_ = params["median"]
        self.iqr_ = params["iqr"]
        return self


def winsorize(
    df: pl.DataFrame,
    columns: List[str],
    lower: float = 0.01,
    upper: float = 0.99,
) -> pl.DataFrame:
    """Clip extreme values to percentile bounds (outlier treatment).

    Args:
        df: Input DataFrame.
        columns: Columns to winsorize.
        lower: Lower percentile (default: 0.01 = 1st percentile).
        upper: Upper percentile (default: 0.99 = 99th percentile).

    Returns:
        DataFrame with winsorized columns.

    Example:
        >>> df = pl.DataFrame({"returns": [-0.5, -0.02, 0.01, 0.02, 0.6]})
        >>> df_winsorized = winsorize(df, ["returns"], lower=0.1, upper=0.9)
    """
    result = df.clone()

    for col in columns:
        lower_bound = df[col].quantile(lower)
        upper_bound = df[col].quantile(upper)

        result = result.with_columns(
            pl.col(col).clip(lower_bound, upper_bound).alias(col)
        )

    return result
