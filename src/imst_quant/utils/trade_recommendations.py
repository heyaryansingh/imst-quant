"""Trade recommendation generator and exporter for automated trading suggestions.

This module provides tools to generate trade recommendations based on signals,
portfolio state, and risk constraints, then export them for review or execution.
"""

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import structlog

logger = structlog.get_logger()


class TradeAction(Enum):
    """Enum for trade actions."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    INCREASE = "INCREASE"


class TradeRecommendation:
    """Represents a single trade recommendation."""

    def __init__(
        self,
        ticker: str,
        action: TradeAction,
        quantity: float,
        signal_strength: float,
        confidence: float,
        rationale: str,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ):
        """Initialize a trade recommendation.

        Args:
            ticker: Asset ticker symbol.
            action: Recommended action (BUY, SELL, etc.).
            quantity: Recommended quantity to trade.
            signal_strength: Signal strength (-1 to 1).
            confidence: Confidence score (0 to 1).
            rationale: Text explanation of the recommendation.
            target_price: Optional target price for the trade.
            stop_loss: Optional stop loss price.
        """
        self.ticker = ticker
        self.action = action
        self.quantity = quantity
        self.signal_strength = signal_strength
        self.confidence = confidence
        self.rationale = rationale
        self.target_price = target_price
        self.stop_loss = stop_loss
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert recommendation to dictionary."""
        return {
            "ticker": self.ticker,
            "action": self.action.value,
            "quantity": self.quantity,
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "target_price": self.target_price,
            "stop_loss": self.stop_loss,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"TradeRecommendation({self.action.value} {self.quantity:.2f} "
            f"{self.ticker}, confidence={self.confidence:.2f})"
        )


def generate_trade_recommendations(
    signals_df: pd.DataFrame,
    current_positions: Dict[str, float],
    max_position_size: float = 0.1,
    min_confidence: float = 0.6,
    signal_col: str = "aggregated_signal",
    confidence_col: str = "confidence",
) -> List[TradeRecommendation]:
    """Generate trade recommendations based on signals and current portfolio.

    Args:
        signals_df: DataFrame with tickers as index, signal and confidence columns.
        current_positions: Dictionary mapping tickers to current position sizes.
        max_position_size: Maximum position size as fraction of portfolio.
        min_confidence: Minimum confidence to generate a recommendation.
        signal_col: Name of the signal column.
        confidence_col: Name of the confidence column.

    Returns:
        List of TradeRecommendation objects.

    Example:
        >>> signals = pd.DataFrame({
        ...     'aggregated_signal': [0.8, -0.7, 0.1],
        ...     'confidence': [0.9, 0.8, 0.5],
        ... }, index=['AAPL', 'MSFT', 'GOOGL'])
        >>> positions = {'AAPL': 0.05, 'MSFT': 0.10}
        >>> recommendations = generate_trade_recommendations(signals, positions)
    """
    recommendations = []

    for ticker in signals_df.index:
        signal = signals_df.loc[ticker, signal_col]
        confidence = signals_df.loc[ticker, confidence_col]

        # Skip low confidence signals
        if confidence < min_confidence:
            continue

        current_position = current_positions.get(ticker, 0.0)

        # Determine action based on signal and current position
        if signal > 0.5:
            # Strong buy signal
            if current_position < max_position_size:
                quantity = max_position_size - current_position
                action = TradeAction.BUY if current_position == 0 else TradeAction.INCREASE
                rationale = (
                    f"Strong positive signal ({signal:.2f}) with high confidence ({confidence:.2f}). "
                    f"Recommend {'opening' if action == TradeAction.BUY else 'increasing'} position."
                )
                recommendations.append(
                    TradeRecommendation(
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        signal_strength=signal,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )

        elif signal < -0.5:
            # Strong sell signal
            if current_position > 0:
                quantity = current_position
                action = TradeAction.SELL
                rationale = (
                    f"Strong negative signal ({signal:.2f}) with high confidence ({confidence:.2f}). "
                    f"Recommend closing position."
                )
                recommendations.append(
                    TradeRecommendation(
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        signal_strength=signal,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )

        elif -0.5 <= signal <= -0.2:
            # Moderate sell signal
            if current_position > max_position_size * 0.5:
                quantity = current_position * 0.5
                action = TradeAction.REDUCE
                rationale = (
                    f"Moderate negative signal ({signal:.2f}). "
                    f"Recommend reducing position by 50%."
                )
                recommendations.append(
                    TradeRecommendation(
                        ticker=ticker,
                        action=action,
                        quantity=quantity,
                        signal_strength=signal,
                        confidence=confidence,
                        rationale=rationale,
                    )
                )

    logger.info(
        "trade_recommendations_generated",
        num_recommendations=len(recommendations),
        buy_count=sum(1 for r in recommendations if r.action == TradeAction.BUY),
        sell_count=sum(1 for r in recommendations if r.action == TradeAction.SELL),
    )

    return recommendations


def export_recommendations_csv(
    recommendations: List[TradeRecommendation],
    output_path: Path,
) -> None:
    """Export trade recommendations to CSV file.

    Args:
        recommendations: List of TradeRecommendation objects.
        output_path: Path to save CSV file.

    Example:
        >>> recs = [TradeRecommendation('AAPL', TradeAction.BUY, 10, 0.8, 0.9, 'Strong signal')]
        >>> export_recommendations_csv(recs, Path('recommendations.csv'))
    """
    if not recommendations:
        logger.warning("no_recommendations_to_export")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([rec.to_dict() for rec in recommendations])
    df.to_csv(output_path, index=False)

    logger.info(
        "recommendations_exported_csv",
        path=str(output_path),
        num_recommendations=len(recommendations),
    )


def export_recommendations_json(
    recommendations: List[TradeRecommendation],
    output_path: Path,
) -> None:
    """Export trade recommendations to JSON file.

    Args:
        recommendations: List of TradeRecommendation objects.
        output_path: Path to save JSON file.

    Example:
        >>> recs = [TradeRecommendation('AAPL', TradeAction.BUY, 10, 0.8, 0.9, 'Strong signal')]
        >>> export_recommendations_json(recs, Path('recommendations.json'))
    """
    if not recommendations:
        logger.warning("no_recommendations_to_export")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = [rec.to_dict() for rec in recommendations]
    import json

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "recommendations_exported_json",
        path=str(output_path),
        num_recommendations=len(recommendations),
    )


def filter_recommendations_by_risk(
    recommendations: List[TradeRecommendation],
    max_total_exposure: float = 1.0,
    current_total_exposure: float = 0.0,
) -> List[TradeRecommendation]:
    """Filter recommendations to stay within risk limits.

    Args:
        recommendations: List of TradeRecommendation objects.
        max_total_exposure: Maximum total portfolio exposure allowed.
        current_total_exposure: Current total portfolio exposure.

    Returns:
        Filtered list of recommendations that fit within risk limits.

    Example:
        >>> recs = [
        ...     TradeRecommendation('AAPL', TradeAction.BUY, 0.3, 0.8, 0.9, 'Strong'),
        ...     TradeRecommendation('MSFT', TradeAction.BUY, 0.4, 0.7, 0.8, 'Strong'),
        ... ]
        >>> filtered = filter_recommendations_by_risk(recs, max_total_exposure=1.0, current_total_exposure=0.5)
    """
    # Sort by confidence * signal_strength descending
    sorted_recs = sorted(
        recommendations,
        key=lambda r: r.confidence * abs(r.signal_strength),
        reverse=True,
    )

    filtered = []
    remaining_capacity = max_total_exposure - current_total_exposure

    for rec in sorted_recs:
        if rec.action in [TradeAction.BUY, TradeAction.INCREASE]:
            if rec.quantity <= remaining_capacity:
                filtered.append(rec)
                remaining_capacity -= rec.quantity
        else:
            # SELL/REDUCE actions don't consume capacity
            filtered.append(rec)

    logger.info(
        "recommendations_filtered_by_risk",
        original_count=len(recommendations),
        filtered_count=len(filtered),
        remaining_capacity=remaining_capacity,
    )

    return filtered


def generate_summary_report(
    recommendations: List[TradeRecommendation],
) -> str:
    """Generate a text summary of trade recommendations.

    Args:
        recommendations: List of TradeRecommendation objects.

    Returns:
        Formatted text summary.

    Example:
        >>> recs = [TradeRecommendation('AAPL', TradeAction.BUY, 10, 0.8, 0.9, 'Strong signal')]
        >>> print(generate_summary_report(recs))
    """
    if not recommendations:
        return "No trade recommendations generated."

    lines = ["=" * 60, "TRADE RECOMMENDATIONS SUMMARY", "=" * 60, ""]

    buy_recs = [r for r in recommendations if r.action in [TradeAction.BUY, TradeAction.INCREASE]]
    sell_recs = [r for r in recommendations if r.action in [TradeAction.SELL, TradeAction.REDUCE]]

    lines.append(f"Total Recommendations: {len(recommendations)}")
    lines.append(f"  - Buy/Increase: {len(buy_recs)}")
    lines.append(f"  - Sell/Reduce: {len(sell_recs)}")
    lines.append("")

    if buy_recs:
        lines.append("BUY/INCREASE RECOMMENDATIONS:")
        lines.append("-" * 60)
        for rec in sorted(buy_recs, key=lambda r: r.confidence, reverse=True):
            lines.append(f"  {rec.ticker:8s} {rec.action.value:10s} {rec.quantity:8.2f}")
            lines.append(f"    Signal: {rec.signal_strength:5.2f} | Confidence: {rec.confidence:5.2f}")
            lines.append(f"    {rec.rationale}")
            lines.append("")

    if sell_recs:
        lines.append("SELL/REDUCE RECOMMENDATIONS:")
        lines.append("-" * 60)
        for rec in sorted(sell_recs, key=lambda r: r.confidence, reverse=True):
            lines.append(f"  {rec.ticker:8s} {rec.action.value:10s} {rec.quantity:8.2f}")
            lines.append(f"    Signal: {rec.signal_strength:5.2f} | Confidence: {rec.confidence:5.2f}")
            lines.append(f"    {rec.rationale}")
            lines.append("")

    lines.append("=" * 60)

    return "\n".join(lines)
