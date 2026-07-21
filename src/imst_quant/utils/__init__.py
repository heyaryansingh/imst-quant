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
- Mean reversion detection (Hurst exponent, ADF test, variance ratio, half-life)
- Sentiment signal conversion (threshold, momentum, contrarian, crossover)
- Scenario analysis (custom stress tests, historical crises, sensitivity sweeps)
- Equity curve analysis (construction, statistics, regimes, rolling CAGR)
- Concentration metrics (Herfindahl, effective N, Gini, Shannon entropy)
- Alpha metrics (decomposition, fundamental law, information ratio)
- Rolling performance (returns, volatility, Sharpe, Sortino, beta, alpha)
- Recovery speed analysis (recovery rate, velocity, pattern classification)
- Covariance shrinkage (Ledoit-Wolf, OAS, identity, custom target estimators)
- Drawdown circuit breaker (tiered alerts, position scaling, cooldown, simulation)
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
from imst_quant.utils.drawdown_recovery import (
    RecoveryAnalysis,
    analyze_recovery_periods,
    estimate_recovery_time,
    recovery_by_depth_bucket,
    recovery_statistics,
    recovery_velocity,
    underwater_analysis as underwater_analysis_detailed,
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
from imst_quant.utils.var_backtesting import (
    christoffersen_conditional_coverage_test,
    christoffersen_independence_test,
    compute_violations,
    kupiec_pof_test,
    var_backtest_summary,
)
from imst_quant.utils.deflated_sharpe import (
    deflated_sharpe_ratio,
    deflated_sharpe_ratio_from_returns,
    estimated_sharpe_ratio_stderr,
    expected_max_sharpe_ratio,
    probabilistic_sharpe_ratio,
)
from imst_quant.utils.position_sizer import (
    PositionSizer,
    risk_parity_weights,
    calculate_leverage,
    dynamic_position_sizing,
)
from imst_quant.utils.portfolio_health import (
    HealthAlert,
    PortfolioHealth,
    PortfolioHealthMonitor,
    generate_health_report,
)
from imst_quant.utils.signal_strength import (
    SignalStrength,
    SignalStrengthAnalyzer,
    compare_signal_strengths,
)
from imst_quant.utils.portfolio_snapshot import (
    HoldingSnapshot,
    PortfolioSnapshot,
    check_alerts,
    compare_snapshots,
    format_snapshot_json,
    format_snapshot_text,
    generate_snapshot,
    snapshot_to_dataframe,
)
from imst_quant.utils.mean_reversion import (
    MeanReversionResult,
    VarianceRatioResult,
    adf_test,
    estimate_half_life,
    find_mean_reverting_pairs,
    generate_mean_reversion_report,
    hurst_exponent,
    rolling_half_life,
    rolling_hurst,
    rolling_variance_ratio,
    test_mean_reversion,
    variance_ratio_test,
)
from imst_quant.utils.cornish_fisher_var import (
    RiskComparison,
    cornish_fisher_cvar,
    cornish_fisher_var,
    jarque_bera_test,
    modified_sharpe_ratio,
    risk_summary,
)
from imst_quant.utils.correlation_regime import (
    CorrelationRegime,
    CorrelationRegimeResult,
    CorrelationStability,
    correlation_divergence,
    correlation_stability,
    detect_correlation_regime,
    eigenvalue_concentration,
    rolling_correlation_matrix,
)
from imst_quant.utils.signal_decay import (
    DecayCurve,
    StalenessResult,
    decay_report,
    detect_signal_staleness,
    measure_signal_decay,
    rolling_signal_ic,
    signal_half_life,
)
from imst_quant.utils.portfolio_turnover import (
    TurnoverCost,
    TurnoverDecomposition,
    TurnoverSummary,
    calculate_turnover,
    estimate_turnover_cost,
    turnover_budget,
    turnover_decomposition,
    turnover_summary,
)
from imst_quant.utils.rebalance_signals import (
    DriftMetrics,
    RebalanceOrder,
    RebalanceSummary,
    calculate_drift,
    calendar_rebalance_signal,
    generate_rebalance_orders,
    rebalance_summary,
    threshold_rebalance_signal,
    volatility_adjusted_threshold,
)
from imst_quant.utils.drawdown_position_scaling import (
    ScalingConfig,
    ScalingState,
    apply_drawdown_scaling,
    calculate_current_drawdown,
    concave_scale_factor,
    convex_scale_factor,
    drawdown_scaling_report,
    get_scaling_state,
    linear_scale_factor,
    step_scale_factor,
)
from imst_quant.utils.sentiment_signals import (
    SentimentSignalConfig,
    sentiment_to_signal,
    sentiment_momentum_signal,
    sentiment_extreme_signal,
    sentiment_crossover_signal,
    composite_sentiment_signal,
    calculate_sentiment_divergence,
)
from imst_quant.utils.scenario_analysis import (
    Scenario,
    ScenarioResult,
    ScenarioAnalysisReport,
    define_scenario,
    apply_scenario,
    run_scenario_analysis,
    historical_scenario_lookup,
    list_historical_scenarios,
    scenario_sensitivity,
    custom_stress_test,
)
from imst_quant.utils.equity_curve import (
    EquityCurveStats,
    PeriodReturn,
    build_equity_curve,
    equity_curve_statistics,
    equity_curve_regimes,
    compare_equity_curves,
    rolling_cagr,
    time_period_returns,
)
from imst_quant.utils.concentration_metrics import (
    herfindahl_index,
    effective_n,
    concentration_ratio,
    gini_coefficient,
    shannon_entropy,
    calculate_all_concentration,
)
from imst_quant.utils.alpha_metrics import (
    AlphaDecomposition,
    AlphaMetrics,
    calculate_fundamental_law_alpha,
    decompose_information_ratio,
)
from imst_quant.utils.rolling_performance import (
    calculate_rolling_returns,
    calculate_rolling_volatility,
    calculate_rolling_sharpe,
    calculate_rolling_sortino,
    calculate_rolling_calmar,
    calculate_rolling_drawdown,
    calculate_rolling_beta,
    calculate_rolling_alpha,
    calculate_rolling_information_ratio,
    calculate_rolling_omega,
    calculate_rolling_win_rate,
    calculate_rolling_profit_factor,
    calculate_rolling_ulcer_index,
    calculate_comprehensive_rolling_metrics,
)
from imst_quant.utils.recovery_speed import (
    RecoveryPattern,
    RecoveryMetrics,
    calculate_recovery_rate,
    analyze_recovery_velocity,
    classify_recovery_pattern,
    recovery_efficiency_score,
)
from imst_quant.utils.covariance_shrinkage import (
    ShrinkageResult,
    ledoit_wolf_shrinkage,
    oas_shrinkage,
    identity_shrinkage,
    custom_target_shrinkage,
    compare_shrinkage_methods,
)
from imst_quant.utils.drawdown_circuit_breaker import (
    CircuitAction,
    CircuitEvent,
    CircuitState,
    DrawdownCircuitBreaker,
    simulate_circuit_breaker,
)
from imst_quant.utils.change_point import (
    analyze_change_points,
    cusum_mean_shift,
    icss_variance_breaks,
)
from imst_quant.utils.conditional_drawdown import (
    analyze_drawdown_risk,
    cdar_ratio,
    conditional_drawdown_at_risk,
    drawdown_at_risk,
    drawdown_series,
)
from imst_quant.utils.higher_moments import (
    analyze_higher_moments,
    cokurtosis,
    coskewness,
    downside_beta,
    upside_beta,
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
    # Drawdown recovery analysis
    "RecoveryAnalysis",
    "analyze_recovery_periods",
    "estimate_recovery_time",
    "recovery_by_depth_bucket",
    "recovery_statistics",
    "recovery_velocity",
    "underwater_analysis_detailed",
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
    # VaR backtesting
    "christoffersen_conditional_coverage_test",
    "christoffersen_independence_test",
    "compute_violations",
    "kupiec_pof_test",
    "var_backtest_summary",
    # Deflated Sharpe ratio
    "deflated_sharpe_ratio",
    "deflated_sharpe_ratio_from_returns",
    "estimated_sharpe_ratio_stderr",
    "expected_max_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    # Position sizer
    "PositionSizer",
    "risk_parity_weights",
    "calculate_leverage",
    "dynamic_position_sizing",
    # Portfolio health
    "HealthAlert",
    "PortfolioHealth",
    "PortfolioHealthMonitor",
    "generate_health_report",
    # Signal strength
    "SignalStrength",
    "SignalStrengthAnalyzer",
    "compare_signal_strengths",
    # Portfolio snapshot
    "HoldingSnapshot",
    "PortfolioSnapshot",
    "check_alerts",
    "compare_snapshots",
    "format_snapshot_json",
    "format_snapshot_text",
    "generate_snapshot",
    "snapshot_to_dataframe",
    # Mean reversion analysis
    "MeanReversionResult",
    "VarianceRatioResult",
    "adf_test",
    "estimate_half_life",
    "find_mean_reverting_pairs",
    "generate_mean_reversion_report",
    "hurst_exponent",
    "rolling_half_life",
    "rolling_hurst",
    "rolling_variance_ratio",
    "test_mean_reversion",
    "variance_ratio_test",
    # Cornish-Fisher VaR
    "RiskComparison",
    "cornish_fisher_cvar",
    "cornish_fisher_var",
    "jarque_bera_test",
    "modified_sharpe_ratio",
    "risk_summary",
    # Correlation regime detection
    "CorrelationRegime",
    "CorrelationRegimeResult",
    "CorrelationStability",
    "correlation_divergence",
    "correlation_stability",
    "detect_correlation_regime",
    "eigenvalue_concentration",
    "rolling_correlation_matrix",
    # Signal decay analysis
    "DecayCurve",
    "StalenessResult",
    "decay_report",
    "detect_signal_staleness",
    "measure_signal_decay",
    "rolling_signal_ic",
    "signal_half_life",
    # Portfolio turnover analysis
    "TurnoverCost",
    "TurnoverDecomposition",
    "TurnoverSummary",
    "calculate_turnover",
    "estimate_turnover_cost",
    "turnover_budget",
    "turnover_decomposition",
    "turnover_summary",
    # Sentiment signals
    "SentimentSignalConfig",
    "sentiment_to_signal",
    "sentiment_momentum_signal",
    "sentiment_extreme_signal",
    "sentiment_crossover_signal",
    "composite_sentiment_signal",
    "calculate_sentiment_divergence",
    # Scenario analysis
    "Scenario",
    "ScenarioResult",
    "ScenarioAnalysisReport",
    "define_scenario",
    "apply_scenario",
    "run_scenario_analysis",
    "historical_scenario_lookup",
    "list_historical_scenarios",
    "scenario_sensitivity",
    "custom_stress_test",
    # Equity curve analysis
    "EquityCurveStats",
    "PeriodReturn",
    "build_equity_curve",
    "equity_curve_statistics",
    "equity_curve_regimes",
    "compare_equity_curves",
    "rolling_cagr",
    "time_period_returns",
    # Concentration metrics
    "herfindahl_index",
    "effective_n",
    "concentration_ratio",
    "gini_coefficient",
    "shannon_entropy",
    "calculate_all_concentration",
    # Alpha metrics
    "AlphaDecomposition",
    "AlphaMetrics",
    "calculate_fundamental_law_alpha",
    "decompose_information_ratio",
    # Rolling performance
    "calculate_rolling_returns",
    "calculate_rolling_volatility",
    "calculate_rolling_sharpe",
    "calculate_rolling_sortino",
    "calculate_rolling_calmar",
    "calculate_rolling_drawdown",
    "calculate_rolling_beta",
    "calculate_rolling_alpha",
    "calculate_rolling_information_ratio",
    "calculate_rolling_omega",
    "calculate_rolling_win_rate",
    "calculate_rolling_profit_factor",
    "calculate_rolling_ulcer_index",
    "calculate_comprehensive_rolling_metrics",
    # Recovery speed
    "RecoveryPattern",
    "RecoveryMetrics",
    "calculate_recovery_rate",
    "analyze_recovery_velocity",
    "classify_recovery_pattern",
    "recovery_efficiency_score",
    # Covariance shrinkage
    "ShrinkageResult",
    "ledoit_wolf_shrinkage",
    "oas_shrinkage",
    "identity_shrinkage",
    "custom_target_shrinkage",
    "compare_shrinkage_methods",
    # Drawdown circuit breaker
    "CircuitAction",
    "CircuitEvent",
    "CircuitState",
    "DrawdownCircuitBreaker",
    "simulate_circuit_breaker",
    # Change-point detection
    "analyze_change_points",
    "cusum_mean_shift",
    "icss_variance_breaks",
    # Conditional drawdown (DaR / CDaR)
    "analyze_drawdown_risk",
    "cdar_ratio",
    "conditional_drawdown_at_risk",
    "drawdown_at_risk",
    "drawdown_series",
    # Higher-moment systematic risk
    "analyze_higher_moments",
    "cokurtosis",
    "coskewness",
    "downside_beta",
    "upside_beta",
]
