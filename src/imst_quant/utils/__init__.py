"""Utility module for IMST-Quant.

This package provides utility functions for:
- Checkpoint management for incremental data crawling
- Risk metrics calculations (Sharpe, Sortino, VaR, Max Drawdown)
- Tail risk analysis (CVaR, Omega ratio, EVT VaR, tail dependency, stress tests)
- Technical indicators (MACD, Bollinger Bands, ATR, ADX, etc.)
- Drawdown analysis (periods, underwater analysis, statistics)
- Market regime detection (volatility, trend, combined regimes)
- Monte Carlo simulation for risk assessment and scenario analysis
- Benchmark comparison and relative performance metrics
- Performance attribution analysis (Brinson, factor-based)
- Trade journal for logging and analyzing trades
- Liquidity analysis (Amihud, spread estimation, market impact)
- Order flow analysis (VPIN, volume imbalance, trade classification)
- Cointegration analysis for pairs trading (Engle-Granger, Kalman filter)
- Portfolio optimization (mean-variance, risk parity, Black-Litterman, HRP)
- Factor analysis (Fama-French exposures, beta decomposition, risk attribution)
- Execution analytics (slippage tracking, fill analysis, implementation shortfall)
- Streak analysis (win/loss streaks, recovery times, gambler's ruin)
- Volatility forecasting (EWMA, GARCH, historical, volatility cone)
- Returns distribution analysis (skewness, kurtosis, normality tests)
- Signal backtesting (quick signal validation, performance metrics)
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
from imst_quant.utils.trade_journal import (
    JournalStatistics,
    TradeEntry,
    TradeJournal,
)
from imst_quant.utils.liquidity_analysis import (
    LiquidityMetrics,
    analyze_liquidity,
    calculate_amihud_illiquidity,
    calculate_roll_spread,
    calculate_volume_profile,
    estimate_market_impact,
    find_illiquid_periods,
)
from imst_quant.utils.order_flow import (
    OrderFlowMetrics,
    analyze_order_flow,
    calculate_vpin,
    classify_trades,
    order_flow_imbalance,
    order_flow_momentum,
    trade_flow_toxicity,
    volume_clock_bars,
    volume_imbalance,
)
from imst_quant.utils.cointegration import (
    CointegrationResult,
    PairsTradingSignal,
    calculate_half_life,
    calculate_hedge_ratio,
    calculate_spread,
    calculate_zscore,
    find_cointegrated_pairs,
    generate_pairs_signal,
    kalman_hedge_ratio,
    rolling_hedge_ratio,
    test_cointegration,
)
from imst_quant.utils.portfolio_optimization import (
    EfficientFrontier,
    OptimizationObjective,
    PortfolioStats,
    black_litterman,
    calculate_efficient_frontier,
    hierarchical_risk_parity,
    mean_variance_optimize,
    minimum_tracking_error,
    risk_parity_optimize,
)
from imst_quant.utils.factor_analysis import (
    FactorExposures,
    RiskDecomposition,
    decompose_risk,
    estimate_factor_exposures,
    factor_attribution,
    generate_synthetic_factors,
    rolling_factor_exposures,
)
from imst_quant.utils.execution_analytics import (
    ExecutedTrade,
    ExecutionMetrics,
    OrderSide,
    OrderType,
    analyze_execution_quality,
    calculate_slippage,
    estimate_expected_slippage,
    generate_execution_report,
    vwap_deviation,
)
from imst_quant.utils.streak_analysis import (
    StreakPeriod,
    StreakStatistics,
    analyze_streaks,
    calculate_gambler_ruin_prob,
    generate_streak_report,
    identify_streaks,
)
from imst_quant.utils.volatility_forecast import (
    VolatilityCone,
    VolatilityForecast,
    compare_volatility_methods,
    ewma_volatility,
    garman_klass_volatility,
    garch_volatility,
    historical_volatility,
    parkinson_volatility,
    simple_garch_volatility,
    volatility_cone,
    volatility_forecast,
    volatility_term_structure,
)
from imst_quant.utils.returns_distribution import (
    DistributionStats,
    NormalityTests,
    TailAnalysis,
    analyze_distribution,
    analyze_tails,
    calculate_moments,
    compare_periods,
    distribution_summary,
    quantile_comparison,
    rolling_kurtosis,
    rolling_skewness,
    test_normality,
)
from imst_quant.utils.signal_backtest import (
    SignalBacktestResult,
    SignalComparison,
    backtest_signal,
    bootstrap_signal,
    combine_signals,
    compare_signals,
    generate_random_signal,
    rolling_signal_performance,
    signal_decay_analysis,
    signal_statistics,
    turnover_analysis,
)
from imst_quant.utils.tail_risk import (
    calculate_all_tail_metrics,
    conditional_var,
    extreme_value_at_risk,
    omega_ratio,
    stress_test_scenarios,
    tail_dependency,
    tail_ratio,
)
from imst_quant.utils.var_calculator import (
    VaRCalculator,
    calculate_portfolio_var,
    stress_test_var,
)
from imst_quant.utils.position_sizer import (
    PositionSizer,
    risk_parity_weights,
    calculate_leverage,
    dynamic_position_sizing,
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
    # Trade journal
    "TradeJournal",
    "TradeEntry",
    "JournalStatistics",
    # Liquidity analysis
    "LiquidityMetrics",
    "analyze_liquidity",
    "calculate_amihud_illiquidity",
    "calculate_roll_spread",
    "calculate_volume_profile",
    "estimate_market_impact",
    "find_illiquid_periods",
    # Order flow analysis
    "OrderFlowMetrics",
    "analyze_order_flow",
    "calculate_vpin",
    "classify_trades",
    "order_flow_imbalance",
    "order_flow_momentum",
    "trade_flow_toxicity",
    "volume_clock_bars",
    "volume_imbalance",
    # Cointegration analysis
    "CointegrationResult",
    "PairsTradingSignal",
    "calculate_half_life",
    "calculate_hedge_ratio",
    "calculate_spread",
    "calculate_zscore",
    "find_cointegrated_pairs",
    "generate_pairs_signal",
    "kalman_hedge_ratio",
    "rolling_hedge_ratio",
    "test_cointegration",
    # Portfolio optimization
    "EfficientFrontier",
    "OptimizationObjective",
    "PortfolioStats",
    "black_litterman",
    "calculate_efficient_frontier",
    "hierarchical_risk_parity",
    "mean_variance_optimize",
    "minimum_tracking_error",
    "risk_parity_optimize",
    # Factor analysis
    "FactorExposures",
    "RiskDecomposition",
    "decompose_risk",
    "estimate_factor_exposures",
    "factor_attribution",
    "generate_synthetic_factors",
    "rolling_factor_exposures",
    # Execution analytics
    "ExecutedTrade",
    "ExecutionMetrics",
    "OrderSide",
    "OrderType",
    "analyze_execution_quality",
    "calculate_slippage",
    "estimate_expected_slippage",
    "generate_execution_report",
    "vwap_deviation",
    # Streak analysis
    "StreakPeriod",
    "StreakStatistics",
    "analyze_streaks",
    "calculate_gambler_ruin_prob",
    "generate_streak_report",
    "identify_streaks",
    # Volatility forecasting
    "VolatilityCone",
    "VolatilityForecast",
    "compare_volatility_methods",
    "ewma_volatility",
    "garman_klass_volatility",
    "garch_volatility",
    "historical_volatility",
    "parkinson_volatility",
    "simple_garch_volatility",
    "volatility_cone",
    "volatility_forecast",
    "volatility_term_structure",
    # Returns distribution analysis
    "DistributionStats",
    "NormalityTests",
    "TailAnalysis",
    "analyze_distribution",
    "analyze_tails",
    "calculate_moments",
    "compare_periods",
    "distribution_summary",
    "quantile_comparison",
    "rolling_kurtosis",
    "rolling_skewness",
    "test_normality",
    # Signal backtesting
    "SignalBacktestResult",
    "SignalComparison",
    "backtest_signal",
    "bootstrap_signal",
    "combine_signals",
    "compare_signals",
    "generate_random_signal",
    "rolling_signal_performance",
    "signal_decay_analysis",
    "signal_statistics",
    "turnover_analysis",
    # Tail risk analysis
    "calculate_all_tail_metrics",
    "conditional_var",
    "extreme_value_at_risk",
    "omega_ratio",
    "stress_test_scenarios",
    "tail_dependency",
    "tail_ratio",
    # VaR calculator
    "VaRCalculator",
    "calculate_portfolio_var",
    "stress_test_var",
    # Position sizer
    "PositionSizer",
    "risk_parity_weights",
    "calculate_leverage",
    "dynamic_position_sizing",
]
