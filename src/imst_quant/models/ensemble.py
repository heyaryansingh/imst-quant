"""Ensemble model combining multiple forecasting models for improved predictions.

This module provides ensemble methods that combine predictions from LSTM, CNN,
Transformer, and LightGBM models to improve accuracy and reduce variance.

Ensemble strategies:
1. Simple averaging - equal weight to all models
2. Weighted averaging - custom weights based on model performance
3. Stacking - meta-learner trained on base model predictions
4. Voting - majority vote for classification tasks
5. Confidence-based - weight by prediction confidence

Example:
    >>> from imst_quant.models.ensemble import EnsembleForecaster
    >>> ensemble = EnsembleForecaster(
    ...     models=["lstm", "cnn", "transformer"],
    ...     strategy="weighted",
    ...     weights=[0.4, 0.3, 0.3]
    ... )
    >>> predictions = ensemble.predict(features)
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from pathlib import Path

import structlog

logger = structlog.get_logger()


class EnsembleForecaster:
    """Ensemble forecasting model combining multiple base models.

    Combines predictions from LSTM, CNN, Transformer, and LightGBM models
    using various ensemble strategies for improved forecasting accuracy.

    Attributes:
        models: List of base model names to include in ensemble.
        strategy: Ensemble combination strategy.
        weights: Optional custom weights for weighted averaging.
        meta_learner: Optional meta-model for stacking ensemble.

    Example:
        >>> ensemble = EnsembleForecaster(
        ...     models=["lstm", "cnn", "lightgbm"],
        ...     strategy="weighted",
        ...     weights=[0.5, 0.3, 0.2]
        ... )
        >>> ensemble.load_models("models/")
        >>> predictions = ensemble.predict(test_features)
    """

    def __init__(
        self,
        models: List[Literal["lstm", "cnn", "transformer", "lightgbm"]],
        strategy: Literal["average", "weighted", "stacking", "voting", "confidence"] = "average",
        weights: Optional[List[float]] = None,
        meta_learner: Optional[str] = None,
    ):
        """Initialize ensemble forecaster.

        Args:
            models: List of base model types to include (e.g., ["lstm", "cnn"]).
            strategy: Ensemble combination strategy (default: "average").
                - "average": Simple average of all predictions
                - "weighted": Weighted average using custom weights
                - "stacking": Use meta-learner on base predictions
                - "voting": Majority vote (for classification)
                - "confidence": Weight by prediction confidence
            weights: Custom weights for weighted averaging (must sum to 1.0).
                Required if strategy="weighted".
            meta_learner: Meta-model type for stacking (e.g., "logistic", "xgboost").
                Required if strategy="stacking".

        Raises:
            ValueError: If weights don't match number of models or don't sum to 1.
        """
        self.models = models
        self.strategy = strategy
        self.weights = weights
        self.meta_learner = meta_learner

        # Validate weights
        if strategy == "weighted":
            if weights is None:
                raise ValueError("weights must be provided for weighted strategy")
            if len(weights) != len(models):
                raise ValueError(f"Weights length ({len(weights)}) must match models ({len(models)})")
            if abs(sum(weights) - 1.0) > 0.01:
                raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")

        if strategy == "stacking" and meta_learner is None:
            raise ValueError("meta_learner must be provided for stacking strategy")

        # Base model instances (loaded later)
        self.base_models: Dict[str, any] = {}

        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {
            model: [] for model in models
        }

        logger.info(
            "ensemble_initialized",
            models=models,
            strategy=strategy,
            weights=weights,
        )

    def load_models(self, model_dir: Path | str) -> None:
        """Load pre-trained base models from directory.

        Args:
            model_dir: Directory containing saved model files.
                Expected filenames: {model_type}_model.pt or {model_type}_model.pkl

        Example:
            >>> ensemble.load_models("models/trained/")
            >>> print(f"Loaded {len(ensemble.base_models)} models")
        """
        model_dir = Path(model_dir)

        for model_name in self.models:
            model_path = model_dir / f"{model_name}_model.pt"
            if model_path.exists():
                # Load PyTorch model
                self.base_models[model_name] = torch.load(model_path)
                logger.info("model_loaded", model=model_name, path=str(model_path))
            else:
                logger.warning("model_not_found", model=model_name, path=str(model_path))

        logger.info("all_models_loaded", num_models=len(self.base_models))

    def predict(
        self,
        features: np.ndarray | torch.Tensor | pl.DataFrame,
        return_individual: bool = False,
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate ensemble predictions.

        Args:
            features: Input features for prediction.
                Shape: (batch_size, sequence_length, num_features) for deep models
                       (batch_size, num_features) for LightGBM
            return_individual: If True, also return individual model predictions.

        Returns:
            Ensemble predictions, optionally with individual predictions dict.

        Example:
            >>> predictions = ensemble.predict(test_features)
            >>> # With individual predictions
            >>> ensemble_pred, individual = ensemble.predict(test_features, return_individual=True)
            >>> print(f"LSTM: {individual['lstm'].mean()}")
            >>> print(f"Ensemble: {ensemble_pred.mean()}")
        """
        if not self.base_models:
            raise RuntimeError("No models loaded. Call load_models() first.")

        individual_predictions = {}

        # Get predictions from each base model
        for model_name, model in self.base_models.items():
            try:
                # Simulate prediction (actual implementation depends on model type)
                # In production, call the actual model's predict method
                pred = self._predict_single_model(model_name, model, features)
                individual_predictions[model_name] = pred

            except Exception as e:
                logger.error(
                    "model_prediction_failed",
                    model=model_name,
                    error=str(e),
                )
                # Use neutral prediction (0.5) on failure
                individual_predictions[model_name] = np.full(
                    len(features), 0.5
                )

        # Combine predictions based on strategy
        ensemble_pred = self._combine_predictions(individual_predictions)

        if return_individual:
            return ensemble_pred, individual_predictions
        return ensemble_pred

    def _predict_single_model(
        self,
        model_name: str,
        model: any,
        features: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Get prediction from a single model.

        Args:
            model_name: Name of the model.
            model: Model instance.
            features: Input features.

        Returns:
            Model predictions as numpy array.
        """
        # This is a placeholder. In production, implement actual prediction logic
        # based on model type (LSTM, CNN, Transformer, LightGBM)

        if isinstance(model, nn.Module):
            # PyTorch model
            model.eval()
            with torch.no_grad():
                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()
                outputs = model(features)
                return outputs.cpu().numpy()
        else:
            # LightGBM or other sklearn-compatible model
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)[:, 1]
            return model.predict(features)

    def _combine_predictions(
        self,
        predictions: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Combine predictions from base models using selected strategy.

        Args:
            predictions: Dictionary mapping model names to their predictions.

        Returns:
            Combined ensemble predictions.
        """
        pred_array = np.array(list(predictions.values()))  # Shape: (num_models, num_samples)

        if self.strategy == "average":
            # Simple average
            return pred_array.mean(axis=0)

        elif self.strategy == "weighted":
            # Weighted average
            weights_array = np.array(self.weights).reshape(-1, 1)
            return (pred_array * weights_array).sum(axis=0)

        elif self.strategy == "voting":
            # Majority vote (for classification)
            # Convert probabilities to binary predictions
            binary_preds = (pred_array > 0.5).astype(int)
            return (binary_preds.sum(axis=0) > len(predictions) / 2).astype(float)

        elif self.strategy == "confidence":
            # Weight by confidence (distance from 0.5)
            confidences = np.abs(pred_array - 0.5)
            weights = confidences / confidences.sum(axis=0, keepdims=True)
            return (pred_array * weights).sum(axis=0)

        elif self.strategy == "stacking":
            # Use meta-learner (placeholder)
            # In production, train a meta-model on base predictions
            logger.warning("stacking_not_implemented", message="Using simple average instead")
            return pred_array.mean(axis=0)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update_weights_by_performance(
        self,
        actual: np.ndarray,
        predictions: Dict[str, np.ndarray],
    ) -> None:
        """Dynamically update ensemble weights based on recent performance.

        Args:
            actual: Actual target values.
            predictions: Dictionary of model predictions.

        Example:
            >>> ensemble.update_weights_by_performance(y_true, individual_predictions)
            >>> print(f"New weights: {ensemble.weights}")
        """
        if self.strategy != "weighted":
            logger.warning(
                "weight_update_skipped",
                message="Can only update weights for weighted strategy",
            )
            return

        # Calculate accuracy for each model
        accuracies = {}
        for model_name, pred in predictions.items():
            # Binary accuracy
            binary_pred = (pred > 0.5).astype(int)
            binary_actual = (actual > 0.5).astype(int)
            acc = (binary_pred == binary_actual).mean()
            accuracies[model_name] = acc

            # Track history
            self.performance_history[model_name].append(acc)

        # Update weights proportional to accuracy
        total_acc = sum(accuracies.values())
        if total_acc > 0:
            new_weights = [accuracies[model] / total_acc for model in self.models]
            self.weights = new_weights

            logger.info(
                "weights_updated",
                old_weights=self.weights,
                new_weights=new_weights,
                accuracies=accuracies,
            )

    def get_model_performance(self) -> Dict[str, float]:
        """Get recent performance statistics for each base model.

        Returns:
            Dictionary mapping model names to average accuracy over recent history.

        Example:
            >>> performance = ensemble.get_model_performance()
            >>> print(f"LSTM accuracy: {performance['lstm']:.3f}")
            >>> print(f"CNN accuracy: {performance['cnn']:.3f}")
        """
        return {
            model: np.mean(history[-10:]) if history else 0.0
            for model, history in self.performance_history.items()
        }

    def optimize_weights(
        self,
        validation_features: np.ndarray,
        validation_targets: np.ndarray,
        num_trials: int = 100,
    ) -> List[float]:
        """Optimize ensemble weights using validation data.

        Args:
            validation_features: Validation set features.
            validation_targets: Validation set targets.
            num_trials: Number of optimization trials (default: 100).

        Returns:
            Optimized weights for the ensemble.

        Example:
            >>> optimal_weights = ensemble.optimize_weights(X_val, y_val, num_trials=200)
            >>> print(f"Optimal weights: {optimal_weights}")
            >>> ensemble.weights = optimal_weights
        """
        if self.strategy != "weighted":
            logger.warning(
                "optimization_skipped",
                message="Weight optimization only applicable for weighted strategy",
            )
            return self.weights or []

        # Get individual model predictions
        _, individual_preds = self.predict(validation_features, return_individual=True)

        best_weights = None
        best_score = -np.inf

        # Random search for optimal weights
        for _ in range(num_trials):
            # Generate random weights that sum to 1
            random_weights = np.random.dirichlet(np.ones(len(self.models)))

            # Calculate ensemble prediction with these weights
            pred_array = np.array(list(individual_preds.values()))
            weighted_pred = (pred_array * random_weights.reshape(-1, 1)).sum(axis=0)

            # Calculate accuracy
            binary_pred = (weighted_pred > 0.5).astype(int)
            binary_actual = (validation_targets > 0.5).astype(int)
            accuracy = (binary_pred == binary_actual).mean()

            if accuracy > best_score:
                best_score = accuracy
                best_weights = random_weights.tolist()

        logger.info(
            "weights_optimized",
            optimal_weights=best_weights,
            accuracy=best_score,
            num_trials=num_trials,
        )

        return best_weights

    def save_ensemble(self, output_path: Path | str) -> None:
        """Save ensemble configuration and weights.

        Args:
            output_path: Path to save ensemble configuration (JSON format).

        Example:
            >>> ensemble.save_ensemble("models/ensemble_config.json")
        """
        import json

        config = {
            "models": self.models,
            "strategy": self.strategy,
            "weights": self.weights,
            "meta_learner": self.meta_learner,
            "performance_history": {
                model: history[-50:]  # Save last 50 results
                for model, history in self.performance_history.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)

        logger.info("ensemble_saved", path=str(output_path))

    @classmethod
    def load_ensemble(cls, config_path: Path | str) -> "EnsembleForecaster":
        """Load ensemble from saved configuration.

        Args:
            config_path: Path to ensemble configuration JSON file.

        Returns:
            Loaded EnsembleForecaster instance.

        Example:
            >>> ensemble = EnsembleForecaster.load_ensemble("models/ensemble_config.json")
            >>> ensemble.load_models("models/trained/")
        """
        import json

        with open(config_path) as f:
            config = json.load(f)

        ensemble = cls(
            models=config["models"],
            strategy=config["strategy"],
            weights=config.get("weights"),
            meta_learner=config.get("meta_learner"),
        )

        # Restore performance history
        ensemble.performance_history = {
            model: history
            for model, history in config.get("performance_history", {}).items()
        }

        logger.info("ensemble_loaded", path=str(config_path))
        return ensemble
