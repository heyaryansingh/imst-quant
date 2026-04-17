"""Utility module for IMST-Quant.

This package provides utility functions for:
- Checkpoint management for incremental data crawling
- Risk metrics calculations (Sharpe, Sortino, VaR, Max Drawdown)
- Technical indicators (MACD, Bollinger Bands, ATR, ADX, etc.)
- Drawdown analysis (periods, underwater analysis, statistics)
- Market regime detection (volatility, trend, combined regimes)
- Monte Carlo simulation for risk assessment and scenario analysis
- Benchmark comparison and relative performance metrics
- Performance attribution analysis (Brinson, factor-based)
"""

from imst_quant.utils.attribution import (
    BrinsonAttribution,
    FactorAttribution,
    PerformanceAttributor,
)
from imst_quant.utils.benchmark import (
    BenchmarkAnalyzer,
    BenchmarkMetrics,
)
from imst_quant.utils.checkpoint import CheckpointManager
from imst_quant.utils.drawdown_analysis import (
    DrawdownPeriod,
    analyze_underwater,
    calculate_drawdown_series,
    drawdown_duration_analysis,
    drawdown_statistics,
    identify_drawdown_periods,
    worst_drawdowns,
)
from imst_quant.utils.regime_detection import (
    MarketRegime,
    TrendRegime,
    VolatilityRegime,
    detect_combined_regime,
    detect_trend_regime,
    detect_volatility_regime,
    estimate_regime_persistence,
    regime_statistics,
    regime_transition_matrix,
)
from imst_quant.utils.risk_metrics import (
    calmar_ratio,
    calculate_all_metrics,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
    value_at_risk,
)
from imst_quant.utils.monte_carlo import (
    MonteCarloSimulator,
    SimulationResult,
)
from imst_quant.utils.technical_indicators import (
    adx,
    atr,
    bollinger_bands,
    cci,
    macd,
    obv,
    stochastic_oscillator,
    vwap,
    williams_r,
)

__all__ = [
    # Checkpoint
    "CheckpointManager",
    # Monte Carlo
    "MonteCarloSimulator",
    "SimulationResult",
    # Benchmark
    "BenchmarkAnalyzer",
    "BenchmarkMetrics",
    # Attribution
    "PerformanceAttributor",
    "BrinsonAttribution",
    "FactorAttribution",
    # Risk metrics
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "value_at_risk",
    "calmar_ratio",
    "calculate_all_metrics",
    # Technical indicators
    "macd",
    "bollinger_bands",
    "atr",
    "adx",
    "stochastic_oscillator",
    "obv",
    "vwap",
    "williams_r",
    "cci",
    # Drawdown analysis
    "DrawdownPeriod",
    "calculate_drawdown_series",
    "identify_drawdown_periods",
    "analyze_underwater",
    "drawdown_statistics",
    "worst_drawdowns",
    "drawdown_duration_analysis",
    # Regime detection
    "VolatilityRegime",
    "TrendRegime",
    "MarketRegime",
    "detect_volatility_regime",
    "detect_trend_regime",
    "detect_combined_regime",
    "regime_statistics",
    "regime_transition_matrix",
    "estimate_regime_persistence",
]
