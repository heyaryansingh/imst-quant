"""Tests for sentiment_signals, scenario_analysis, and equity_curve utilities."""

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Sentiment signals tests
# ---------------------------------------------------------------------------
from imst_quant.utils.sentiment_signals import (
    SentimentSignalConfig,
    sentiment_to_signal,
    sentiment_momentum_signal,
    sentiment_extreme_signal,
    sentiment_crossover_signal,
    composite_sentiment_signal,
    calculate_sentiment_divergence,
)


class TestSentimentToSignal:
    """Tests for basic sentiment threshold signal conversion."""

    @pytest.fixture
    def sentiment_df(self):
        return pl.DataFrame(
            {
                "sentiment": [0.5, 0.1, -0.3, -0.1, 0.8, -0.9, 0.0],
            }
        )

    def test_default_thresholds(self, sentiment_df):
        result = sentiment_to_signal(sentiment_df)
        signals = result["sentiment_signal"].to_list()
        # 0.5 > 0.2 -> 1, 0.1 < 0.2 -> 0, -0.3 < -0.2 -> -1, etc.
        assert signals[0] == 1
        assert signals[1] == 0
        assert signals[2] == -1
        assert signals[3] == 0
        assert signals[4] == 1
        assert signals[5] == -1
        assert signals[6] == 0

    def test_custom_thresholds(self, sentiment_df):
        config = SentimentSignalConfig(bullish_threshold=0.4, bearish_threshold=-0.4)
        result = sentiment_to_signal(sentiment_df, config=config)
        signals = result["sentiment_signal"].to_list()
        assert signals[0] == 1  # 0.5 > 0.4
        assert signals[1] == 0  # 0.1 in dead zone
        assert signals[2] == 0  # -0.3 > -0.4, still dead zone

    def test_strength_column(self, sentiment_df):
        result = sentiment_to_signal(sentiment_df)
        strengths = result["sentiment_strength"].to_list()
        assert strengths[0] == pytest.approx(0.5)
        assert strengths[5] == pytest.approx(0.9)

    def test_confidence_filtering(self):
        df = pl.DataFrame(
            {
                "sentiment": [0.5, 0.6, -0.5],
                "confidence": [0.8, 0.1, 0.9],
            }
        )
        config = SentimentSignalConfig(min_confidence=0.5)
        result = sentiment_to_signal(df, confidence_col="confidence", config=config)
        signals = result["sentiment_signal"].to_list()
        assert signals[0] == 1  # high confidence
        assert signals[1] == 0  # filtered out by low confidence
        assert signals[2] == -1  # high confidence


class TestSentimentMomentum:
    """Tests for sentiment momentum signals."""

    def test_momentum_detection(self):
        # Sentiment rising sharply
        df = pl.DataFrame({"sentiment": [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]})
        result = sentiment_momentum_signal(df, window=3, momentum_threshold=0.1)
        # Momentum at index 3 = 0.2 - 0.0 = 0.2 >= 0.1 -> bullish
        momenta = result["sentiment_momentum"].to_list()
        assert momenta[3] == pytest.approx(0.2)

    def test_momentum_signal_values(self):
        df = pl.DataFrame({"sentiment": [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]})
        result = sentiment_momentum_signal(df, window=2, momentum_threshold=0.05)
        signals = result["sentiment_momentum_signal"].to_list()
        # Each step drops by 0.1, momentum = -0.1 < -0.05 -> bearish
        for s in signals[2:]:
            assert s == -1


class TestSentimentExtreme:
    """Tests for extreme sentiment contrarian signals."""

    def test_extreme_high(self):
        df = pl.DataFrame({"sentiment": [0.9, 0.5, -0.9, 0.0]})
        result = sentiment_extreme_signal(df)
        signals = result["contrarian_signal"].to_list()
        assert signals[0] == -1  # extreme high -> contrarian sell
        assert signals[2] == 1  # extreme low -> contrarian buy
        assert signals[1] == 0  # normal range
        assert signals[3] == 0

    def test_extreme_flag(self):
        df = pl.DataFrame({"sentiment": [0.85, 0.3]})
        result = sentiment_extreme_signal(df)
        extremes = result["sentiment_extreme"].to_list()
        assert extremes[0] is True
        assert extremes[1] is False


class TestSentimentCrossover:
    """Tests for sentiment MA crossover signals."""

    def test_crossover_columns_exist(self):
        df = pl.DataFrame({"sentiment": np.random.randn(20).tolist()})
        result = sentiment_crossover_signal(df, fast_window=3, slow_window=5)
        assert "sentiment_fast_ma" in result.columns
        assert "sentiment_slow_ma" in result.columns
        assert "crossover_signal" in result.columns

    def test_crossover_values(self):
        # Construct a series that clearly crosses over
        sentiment = [-0.5] * 10 + [0.5] * 10
        df = pl.DataFrame({"sentiment": sentiment})
        result = sentiment_crossover_signal(df, fast_window=3, slow_window=7)
        signals = result["crossover_signal"].to_list()
        # Should have at least one bullish crossover (1) around index 10-12
        assert 1 in signals


class TestCompositeSentimentSignal:
    """Tests for composite sentiment signal combination."""

    def test_composite_columns(self):
        df = pl.DataFrame({"sentiment": np.linspace(-1, 1, 20).tolist()})
        result = composite_sentiment_signal(df)
        assert "composite_sentiment_score" in result.columns
        assert "composite_sentiment_signal" in result.columns
        assert "sentiment_signal" in result.columns
        assert "sentiment_momentum_signal" in result.columns

    def test_composite_with_custom_weights(self):
        df = pl.DataFrame({"sentiment": [0.5] * 15})
        weights = {"threshold": 1.0, "momentum": 0.0, "contrarian": 0.0, "crossover": 0.0}
        result = composite_sentiment_signal(df, weights=weights)
        # With only threshold weight, composite should follow threshold signal
        composites = result["composite_sentiment_signal"].to_list()
        assert all(c == 1 for c in composites)


class TestSentimentDivergence:
    """Tests for sentiment-price divergence detection."""

    def test_divergence_columns(self):
        df = pl.DataFrame(
            {
                "sentiment": np.linspace(-0.5, 0.5, 20).tolist(),
                "returns": np.linspace(0.01, -0.01, 20).tolist(),
            }
        )
        result = calculate_sentiment_divergence(df, window=5)
        assert "sentiment_direction" in result.columns
        assert "price_direction" in result.columns
        assert "sentiment_price_divergence" in result.columns
        assert "divergence_signal" in result.columns


# ---------------------------------------------------------------------------
# Scenario analysis tests
# ---------------------------------------------------------------------------
from imst_quant.utils.scenario_analysis import (
    Scenario,
    ScenarioResult,
    define_scenario,
    apply_scenario,
    run_scenario_analysis,
    historical_scenario_lookup,
    list_historical_scenarios,
    scenario_sensitivity,
    custom_stress_test,
)


class TestDefineScenario:
    """Tests for scenario definition."""

    def test_basic_creation(self):
        s = define_scenario("test", {"SPY": -0.10})
        assert s.name == "test"
        assert s.shocks["SPY"] == -0.10

    def test_probability_clamped(self):
        s = define_scenario("test", {}, probability=1.5)
        assert s.probability == 1.0
        s2 = define_scenario("test", {}, probability=-0.5)
        assert s2.probability == 0.0


class TestApplyScenario:
    """Tests for applying scenarios to portfolios."""

    def test_simple_crash(self):
        weights = {"SPY": 0.6, "TLT": 0.4}
        scenario = define_scenario("crash", {"SPY": -0.20, "TLT": 0.05})
        result = apply_scenario(weights, scenario)
        expected = 0.6 * (-0.20) + 0.4 * 0.05
        assert result.portfolio_impact == pytest.approx(expected)
        assert result.worst_asset == "SPY"
        assert result.best_asset == "TLT"

    def test_missing_asset_shock(self):
        weights = {"SPY": 0.5, "AAPL": 0.5}
        scenario = define_scenario("partial", {"SPY": -0.10})
        result = apply_scenario(weights, scenario)
        # AAPL has no shock, so 0.0
        expected = 0.5 * (-0.10) + 0.5 * 0.0
        assert result.portfolio_impact == pytest.approx(expected)


class TestRunScenarioAnalysis:
    """Tests for running multiple scenarios."""

    def test_multiple_scenarios(self):
        weights = {"SPY": 0.6, "TLT": 0.4}
        scenarios = [
            define_scenario("crash", {"SPY": -0.30, "TLT": 0.10}, probability=0.1),
            define_scenario("boom", {"SPY": 0.20, "TLT": -0.05}, probability=0.3),
        ]
        report = run_scenario_analysis(weights, scenarios)
        assert len(report.results) == 2
        assert report.worst_case.scenario_name == "crash"
        assert isinstance(report.expected_loss, float)

    def test_no_probabilities(self):
        weights = {"SPY": 1.0}
        scenarios = [
            define_scenario("a", {"SPY": -0.10}),
            define_scenario("b", {"SPY": -0.20}),
        ]
        report = run_scenario_analysis(weights, scenarios)
        # Expected loss should be simple average
        assert report.expected_loss == pytest.approx(-0.15)


class TestHistoricalScenarios:
    """Tests for historical scenario lookup."""

    def test_known_scenario(self):
        s = historical_scenario_lookup("2008_financial_crisis")
        assert s is not None
        assert "SPY" in s.shocks
        assert s.shocks["SPY"] < 0

    def test_unknown_scenario(self):
        s = historical_scenario_lookup("nonexistent")
        assert s is None

    def test_list_scenarios(self):
        names = list_historical_scenarios()
        assert len(names) >= 4
        assert "2020_covid_crash" in names


class TestScenarioSensitivity:
    """Tests for scenario sensitivity sweep."""

    def test_sensitivity_output(self):
        weights = {"SPY": 0.6, "TLT": 0.4}
        base = define_scenario("base", {"SPY": -0.10, "TLT": 0.05})
        results = scenario_sensitivity(weights, base, "SPY", steps=10)
        assert len(results) == 10
        # Each result is (shock_value, portfolio_impact)
        for shock_val, impact in results:
            assert isinstance(shock_val, float)
            assert isinstance(impact, float)


class TestCustomStressTest:
    """Tests for custom stress test with correlation amplification."""

    def test_amplified_losses(self):
        weights = {"SPY": 0.6, "TLT": 0.4}
        shocks = {"SPY": -0.10, "TLT": 0.05}
        result = custom_stress_test(weights, shocks, correlation_multiplier=2.0)
        # SPY shock amplified: -0.10 * 2 = -0.20
        # TLT hedge reduced: 0.05 / 2 = 0.025
        expected = 0.6 * (-0.20) + 0.4 * 0.025
        assert result.portfolio_impact == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Equity curve tests
# ---------------------------------------------------------------------------
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


class TestBuildEquityCurve:
    """Tests for equity curve construction."""

    def test_basic_curve(self):
        returns = pl.Series("returns", [0.01, -0.005, 0.02])
        curve = build_equity_curve(returns, initial_capital=10000.0)
        assert len(curve) == 3
        assert "equity" in curve.columns
        assert "drawdown_pct" in curve.columns
        # First period: 10000 * 1.01 = 10100
        assert curve["equity"][0] == pytest.approx(10100.0)

    def test_peak_tracking(self):
        returns = pl.Series("returns", [0.10, -0.05, 0.02])
        curve = build_equity_curve(returns, initial_capital=1000.0)
        peaks = curve["peak"].to_list()
        # Peak should be monotonically non-decreasing
        for i in range(1, len(peaks)):
            assert peaks[i] >= peaks[i - 1]

    def test_drawdown_negative(self):
        returns = pl.Series("returns", [0.10, -0.20])
        curve = build_equity_curve(returns, initial_capital=1000.0)
        # After 10% gain then 20% loss, should be in drawdown
        assert curve["drawdown_pct"][1] < 0


class TestEquityCurveStatistics:
    """Tests for equity curve statistics."""

    @pytest.fixture
    def positive_returns(self):
        np.random.seed(42)
        return pl.Series("returns", (np.random.normal(0.001, 0.01, 252)).tolist())

    def test_stats_structure(self, positive_returns):
        stats = equity_curve_statistics(positive_returns)
        assert isinstance(stats, EquityCurveStats)
        assert stats.num_periods == 252
        assert stats.positive_periods + stats.negative_periods <= 252

    def test_cagr_positive(self, positive_returns):
        stats = equity_curve_statistics(positive_returns)
        # With positive mean return, CAGR should be positive
        assert stats.cagr > 0

    def test_empty_returns(self):
        stats = equity_curve_statistics(pl.Series("returns", []))
        assert stats.total_return == 0.0
        assert stats.num_periods == 0

    def test_max_drawdown_negative(self):
        returns = pl.Series("returns", [0.05, -0.10, -0.10, 0.05])
        stats = equity_curve_statistics(returns)
        assert stats.max_drawdown < 0


class TestEquityCurveRegimes:
    """Tests for equity curve regime labeling."""

    def test_regime_labels(self):
        returns = pl.Series("returns", [0.01, 0.01, -0.10, -0.05, 0.02, 0.03])
        result = equity_curve_regimes(returns, drawdown_threshold=-0.05)
        regimes = result["regime"].to_list()
        assert "growth" in regimes
        assert "drawdown" in regimes

    def test_regime_duration(self):
        returns = pl.Series("returns", [0.01] * 5)
        result = equity_curve_regimes(returns)
        durations = result["regime_duration"].to_list()
        # All growth, duration should increment
        assert durations == [1, 2, 3, 4, 5]


class TestCompareEquityCurves:
    """Tests for multi-strategy comparison."""

    def test_comparison(self):
        np.random.seed(42)
        strategies = {
            "aggressive": pl.Series("r", np.random.normal(0.002, 0.03, 100).tolist()),
            "conservative": pl.Series("r", np.random.normal(0.001, 0.01, 100).tolist()),
        }
        results = compare_equity_curves(strategies)
        assert "aggressive" in results
        assert "conservative" in results
        assert isinstance(results["aggressive"], EquityCurveStats)


class TestRollingCagr:
    """Tests for rolling CAGR calculation."""

    def test_output_length(self):
        returns = pl.Series("returns", [0.01] * 50)
        result = rolling_cagr(returns, window=20)
        assert len(result) == 50

    def test_nan_initial(self):
        returns = pl.Series("returns", [0.01] * 50)
        result = rolling_cagr(returns, window=20)
        # First 19 values should be NaN
        for i in range(19):
            assert np.isnan(result[i])
        # Value at index 19 should be a number
        assert not np.isnan(result[19])
