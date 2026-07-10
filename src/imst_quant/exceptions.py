"""Custom exception hierarchy for IMST-Quant.

Provides structured error types for the data pipeline, model training,
trading execution, and API interaction layers. Using specific exception
types improves error handling, logging, and debugging across the system.

Exception Hierarchy:
    IMSTError
    ├── DataPipelineError
    │   ├── IngestionError
    │   ├── TransformationError
    │   └── ValidationError
    ├── ModelError
    │   ├── TrainingError
    │   └── PredictionError
    ├── TradingError
    │   ├── SignalError
    │   ├── ExecutionError
    │   └── PositionError
    └── ConfigurationError

Example:
    >>> from imst_quant.exceptions import IngestionError
    >>> raise IngestionError("Reddit API rate limit exceeded", source="reddit")
"""


class IMSTError(Exception):
    """Base exception for all IMST-Quant errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dict of structured context for logging.
    """

    def __init__(self, message: str, **details):
        self.message = message
        self.details = details
        super().__init__(message)

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


# ---------------------------------------------------------------------------
# Data pipeline errors
# ---------------------------------------------------------------------------


class DataPipelineError(IMSTError):
    """Base error for data pipeline operations (ingestion, transform, storage)."""

    pass


class IngestionError(DataPipelineError):
    """Error during data ingestion from external sources.

    Raised when fetching data from Reddit (PRAW), yfinance, or CCXT fails
    due to API errors, rate limits, authentication issues, or network problems.
    """

    def __init__(self, message: str, source: str = "unknown", **details):
        super().__init__(message, source=source, **details)


class TransformationError(DataPipelineError):
    """Error during data transformation between pipeline layers.

    Raised when converting between raw/bronze/silver/gold layers fails
    due to schema mismatches, encoding issues, or data corruption.
    """

    def __init__(self, message: str, layer: str = "unknown", **details):
        super().__init__(message, layer=layer, **details)


class ValidationError(DataPipelineError):
    """Error when data fails quality or schema validation checks.

    Raised when data does not conform to expected schemas, contains
    invalid values, or fails integrity checks.
    """

    def __init__(self, message: str, column: str | None = None, **details):
        super().__init__(message, column=column, **details)


# ---------------------------------------------------------------------------
# Model errors
# ---------------------------------------------------------------------------


class ModelError(IMSTError):
    """Base error for ML model operations."""

    pass


class TrainingError(ModelError):
    """Error during model training.

    Raised when training fails due to data issues (NaN features, shape
    mismatches), convergence problems, or resource exhaustion.
    """

    def __init__(self, message: str, model_type: str = "unknown", **details):
        super().__init__(message, model_type=model_type, **details)


class PredictionError(ModelError):
    """Error during model inference / prediction.

    Raised when a trained model fails to produce predictions due to
    input format issues, missing features, or model corruption.
    """

    def __init__(self, message: str, model_type: str = "unknown", **details):
        super().__init__(message, model_type=model_type, **details)


# ---------------------------------------------------------------------------
# Trading errors
# ---------------------------------------------------------------------------


class TradingError(IMSTError):
    """Base error for trading operations."""

    pass


class SignalError(TradingError):
    """Error in trading signal generation or validation.

    Raised when signal computation produces invalid results, encounters
    missing data, or detects logical inconsistencies.
    """

    def __init__(self, message: str, signal_type: str = "unknown", **details):
        super().__init__(message, signal_type=signal_type, **details)


class ExecutionError(TradingError):
    """Error during trade execution.

    Raised when order placement, fill tracking, or execution quality
    measurement fails.
    """

    def __init__(self, message: str, asset: str | None = None, **details):
        super().__init__(message, asset=asset, **details)


class PositionError(TradingError):
    """Error in position management.

    Raised when position sizing, tracking, or reconciliation encounters
    invalid states (negative quantities, exceeding limits, etc.).
    """

    def __init__(self, message: str, asset: str | None = None, **details):
        super().__init__(message, asset=asset, **details)


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigurationError(IMSTError):
    """Error in system configuration.

    Raised when required environment variables are missing, config files
    are malformed, or settings are invalid.
    """

    def __init__(self, message: str, setting: str | None = None, **details):
        super().__init__(message, setting=setting, **details)
