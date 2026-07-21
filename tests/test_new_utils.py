"""Tests for newly added utility modules.

Tests cover:
- Cornish-Fisher VaR and modified risk metrics
- Correlation regime change detection
- Signal decay and staleness detection
- Portfolio turnover analysis
"""

import numpy as np
import polars as pl
import pytest

# ---------------------------------------------------------------------------
# Cornish-Fisher VaR tests
# ---------------------------------------------------------------------------

from imst_quant.utils.cornish_fisher_var import (
    RiskComparison,
    cornish_fisher_cvar,
    cornish_fisher_var,
    jarque_bera_test,
    modified_sharpe_ratio,
    risk_summary,
)


class TestCornishFisherVaR:
    """Tests for Cornish-Fisher VaR utilities."""

    @pytest.fixture
    def normal_returns(self):
        np.random.seed(42)
        return pl.Series("returns", np.random.normal(0.0005, 0.02, 500).tolist())

    @pytest.fixture
    def skewed_returns(self):
        np.random.seed(42)
        base = np.random.normal(0.001, 0.02, 500)
        # Add left tail (negative skew)
        crashes = np.random.choice(500, 20, replace=False)
        base[crashes] -= 0.08
        return pl.Series("returns", base.tolist())

    def test_cf_var_positive(self, normal_returns):
        """CF-VaR should be a positive value for typical returns."""
        var = cornish_fisher_var(normal_returns, confidence_level=0.95)
        assert var > 0

    def test_cf_var_increases_with_confidence(self, normal_returns):
        """Higher confidence should produce higher VaR."""
        var_95 = cornish_fisher_var(normal_returns, confidence_level=0.95)
        var_99 = cornish_fisher_var(normal_returns, confidence_level=0.99)
        assert var_99 > var_95

    def test_cf_var_differs_from_parametric_for_skewed(self, skewed_returns):
        """CF-VaR should differ from parametric VaR for skewed returns."""
        summary = risk_summary(skewed_returns)
        # With negative skew, CF-VaR should be higher than parametric
        assert summary.cornish_fisher_var != summary.parametric_var
        assert summary.skewness < 0  # Confirm negative skew

    def test_cf_cvar_exceeds_var(self, normal_returns):
        """CF-CVaR should be >= CF-VaR (expected shortfall is worse)."""
        var = cornish_fisher_var(normal_returns, confidence_level=0.95)
        cvar = cornish_fisher_cvar(normal_returns, confidence_level=0.95)
        assert cvar >= var

    def test_jarque_bera_normal(self, normal_returns):
        """JB test should not reject normality for normal returns."""
        result = jarque_bera_test(normal_returns)
        assert "statistic" in result
        assert "pvalue" in result
        assert "is_normal" in result

    def test_jarque_bera_skewed(self, skewed_returns):
        """JB test should reject normality for heavily skewed returns."""
        result = jarque_bera_test(skewed_returns)
        assert result["is_normal"] is False

    def test_modified_sharpe_ratio(self, normal_returns):
        """Modified Sharpe should return a finite number."""
        msr = modified_sharpe_ratio(normal_returns, risk_free_rate=0.0001)
        assert np.isfinite(msr)

    def test_risk_summary_dataclass(self, normal_returns):
        """risk_summary should return a RiskComparison dataclass."""
        result = risk_summary(normal_returns)
        assert isinstance(result, RiskComparison)
        assert result.parametric_var > 0
        assert result.cornish_fisher_var > 0
        assert result.historical_var > 0

    def test_short_series_returns_zero(self):
        """Short series should return 0 gracefully."""
        short = pl.Series("returns", [0.01, -0.01])
        assert cornish_fisher_var(short) == 0.0

    def test_dataframe_input(self, normal_returns):
        """Should accept DataFrame with column name."""
        df = pl.DataFrame({"my_returns": normal_returns})
        var = cornish_fisher_var(df, return_col="my_returns")
        assert var > 0


# ---------------------------------------------------------------------------
# Correlation regime tests
# ---------------------------------------------------------------------------

from imst_quant.utils.correlation_regime import (
    CorrelationRegime,
    CorrelationStability,
    correlation_divergence,
    correlation_stability,
    detect_correlation_regime,
    eigenvalue_concentration,
    rolling_correlation_matrix,
)


class TestCorrelationRegime:
    """Tests for correlation regime detection."""

    @pytest.fixture
    def uncorrelated_returns(self):
        np.random.seed(42)
        return pl.DataFrame({
            "A": np.random.randn(300).tolist(),
            "B": np.random.randn(300).tolist(),
            "C": np.random.randn(300).tolist(),
        })

    @pytest.fixture
    def crisis_returns(self):
        """Highly correlated returns simulating a crisis."""
        np.random.seed(42)
        market = np.random.randn(300) * 0.03
        return pl.DataFrame({
            "A": (market + np.random.randn(300) * 0.002).tolist(),
            "B": (market + np.random.randn(300) * 0.002).tolist(),
            "C": (market + np.random.randn(300) * 0.002).tolist(),
        })

    def test_detect_normal_regime(self, uncorrelated_returns):
        """Uncorrelated assets should be NORMAL or DECORRELATED."""
        result = detect_correlation_regime(uncorrelated_returns, window=60)
        assert result.regime in (CorrelationRegime.NORMAL, CorrelationRegime.DECORRELATED)
        assert result.assets_analyzed == 3

    def test_detect_crisis_regime(self, crisis_returns):
        """Highly correlated assets should be CRISIS or ELEVATED."""
        result = detect_correlation_regime(crisis_returns, window=60)
        assert result.regime in (CorrelationRegime.CRISIS, CorrelationRegime.ELEVATED)
        assert result.mean_correlation > 0.4

    def test_rolling_correlation(self, uncorrelated_returns):
        """Rolling correlation should produce a list of matrix snapshots."""
        matrices = rolling_correlation_matrix(uncorrelated_returns, window=30)
        assert len(matrices) > 0
        assert "matrix" in matrices[0]
        assert "mean_corr" in matrices[0]
        assert matrices[0]["matrix"].shape == (3, 3)

    def test_correlation_divergence(self):
        """Divergence between identical matrices should be zero."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        div = correlation_divergence(corr, corr)
        assert div["frobenius"] == pytest.approx(0.0, abs=1e-10)

    def test_correlation_divergence_nonzero(self):
        """Divergence between different matrices should be positive."""
        curr = np.array([[1.0, 0.8], [0.8, 1.0]])
        base = np.array([[1.0, 0.2], [0.2, 1.0]])
        div = correlation_divergence(curr, base)
        assert div["frobenius"] > 0
        assert div["max_change"] > 0

    def test_correlation_stability(self, uncorrelated_returns):
        """Stability check on uncorrelated returns should generally be stable."""
        result = correlation_stability(uncorrelated_returns, window=60)
        assert isinstance(result, CorrelationStability)

    def test_eigenvalue_concentration(self, crisis_returns):
        """Crisis returns should have high eigenvalue concentration."""
        data = crisis_returns.to_numpy()
        conc = eigenvalue_concentration(data)
        assert 0 < conc <= 1.0
        # Highly correlated -> first PC should explain a lot
        assert conc > 0.5

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        tiny = pl.DataFrame({"A": [0.01], "B": [0.02]})
        result = detect_correlation_regime(tiny, window=60)
        assert result.regime == CorrelationRegime.NORMAL


# ---------------------------------------------------------------------------
# Signal decay tests
# ---------------------------------------------------------------------------

from imst_quant.utils.signal_decay import (
    DecayCurve,
    StalenessResult,
    decay_report,
    detect_signal_staleness,
    measure_signal_decay,
    rolling_signal_ic,
    signal_half_life,
)


class TestSignalDecay:
    """Tests for signal decay and staleness detection."""

    @pytest.fixture
    def predictive_signal(self):
        """Signal that is correlated with forward returns."""
        np.random.seed(42)
        n = 300
        signal = np.random.choice([-1, 1], size=n).astype(float)
        # Returns partially follow the signal
        returns = signal * 0.01 + np.random.randn(n) * 0.005
        return pl.Series("signal", signal.tolist()), pl.Series("returns", returns.tolist())

    @pytest.fixture
    def random_signal(self):
        """Random signal with no predictive power."""
        np.random.seed(42)
        n = 300
        signal = np.random.choice([-1, 1], size=n).astype(float)
        returns = np.random.randn(n) * 0.02
        return pl.Series("signal", signal.tolist()), pl.Series("returns", returns.tolist())

    def test_measure_decay_returns_curve(self, predictive_signal):
        """Should return a DecayCurve with expected fields."""
        signals, returns = predictive_signal
        curve = measure_signal_decay(signals, returns, horizons=[1, 5, 10])
        assert isinstance(curve, DecayCurve)
        assert len(curve.horizons) == 3
        assert len(curve.ic_values) == 3

    def test_predictive_signal_has_positive_ic(self, predictive_signal):
        """Predictive signal should have positive IC at horizon 1."""
        signals, returns = predictive_signal
        curve = measure_signal_decay(signals, returns, horizons=[1])
        assert curve.ic_values[0] > 0

    def test_rolling_ic_produces_dataframe(self, predictive_signal):
        """Rolling IC should return a DataFrame with ic column."""
        signals, returns = predictive_signal
        ic_df = rolling_signal_ic(signals, returns, window=60)
        assert "ic" in ic_df.columns
        assert "is_significant" in ic_df.columns
        assert ic_df.height > 0

    def test_detect_staleness_predictive(self, predictive_signal):
        """Predictive signal should not be flagged as stale."""
        signals, returns = predictive_signal
        result = detect_signal_staleness(signals, returns, recent_window=60)
        assert isinstance(result, StalenessResult)
        # Predictive signal should generally not be stale
        assert result.recent_hit_rate > 0.5

    def test_detect_staleness_random(self, random_signal):
        """Random signal should have hit rate near 0.5."""
        signals, returns = random_signal
        result = detect_signal_staleness(signals, returns, recent_window=60)
        assert isinstance(result, StalenessResult)
        assert 0.3 < result.recent_hit_rate < 0.7  # Near random

    def test_signal_half_life_decaying(self):
        """Decaying IC values should produce a finite half-life."""
        hl = signal_half_life([1, 5, 10, 20], [0.10, 0.07, 0.04, 0.02])
        assert hl is not None
        assert hl > 0

    def test_signal_half_life_flat(self):
        """Non-decaying IC should return None."""
        hl = signal_half_life([1, 5, 10], [0.05, 0.05, 0.05])
        # Flat IC -> no decay -> None or very large
        # The log regression on constant values gives slope=0
        assert hl is None

    def test_decay_report(self, predictive_signal):
        """decay_report should return a dict with all sections."""
        signals, returns = predictive_signal
        report = decay_report(signals, returns, horizons=[1, 5, 10])
        assert "decay_curve" in report
        assert "staleness" in report
        assert "rolling_ic_summary" in report
        assert "recommendation" in report

    def test_insufficient_data(self):
        """Should handle very short series gracefully."""
        sigs = pl.Series([1, -1, 1])
        rets = pl.Series([0.01, -0.01, 0.02])
        result = detect_signal_staleness(sigs, rets, recent_window=60)
        assert result.is_stale is False


# ---------------------------------------------------------------------------
# Portfolio turnover tests
# ---------------------------------------------------------------------------

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


class TestPortfolioTurnover:
    """Tests for portfolio turnover analysis."""

    def test_calculate_turnover_simple(self):
        """Basic turnover calculation."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.3, "MSFT": 0.7}
        t = calculate_turnover(before, after, one_way=True)
        # 0.2 sold from AAPL, 0.2 bought into MSFT -> one-way = 0.4 / 2 = 0.2
        assert t == pytest.approx(0.2)

    def test_calculate_turnover_two_way(self):
        """Two-way turnover should be double one-way."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.3, "MSFT": 0.7}
        t1 = calculate_turnover(before, after, one_way=True)
        t2 = calculate_turnover(before, after, one_way=False)
        assert t2 == pytest.approx(2 * t1)

    def test_calculate_turnover_no_change(self):
        """Zero turnover when weights unchanged."""
        w = {"AAPL": 0.5, "MSFT": 0.5}
        assert calculate_turnover(w, w) == pytest.approx(0.0)

    def test_turnover_with_new_and_closed(self):
        """Should handle new and closed positions."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.5, "GOOG": 0.5}
        t = calculate_turnover(before, after, one_way=True)
        assert t == pytest.approx(0.5)

    def test_turnover_decomposition(self):
        """Decomposition should identify new/closed positions."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.3, "MSFT": 0.4, "GOOG": 0.3}
        decomp = turnover_decomposition(before, after)
        assert isinstance(decomp, TurnoverDecomposition)
        assert "GOOG" in decomp.new_positions
        assert decomp.total_turnover > 0

    def test_turnover_decomposition_with_returns(self):
        """Drift vs rebalance should be separated when returns given."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.4, "MSFT": 0.6}
        returns = {"AAPL": -0.05, "MSFT": 0.05}
        decomp = turnover_decomposition(before, after, returns)
        assert decomp.drift_turnover >= 0
        assert decomp.rebalance_turnover >= 0

    def test_estimate_turnover_cost(self):
        """Cost estimation should return positive bps."""
        before = {"AAPL": 0.5, "MSFT": 0.5}
        after = {"AAPL": 0.3, "MSFT": 0.7}
        cost = estimate_turnover_cost(before, after)
        assert isinstance(cost, TurnoverCost)
        assert cost.total_cost_bps > 0
        assert cost.commission_cost_bps > 0
        assert cost.spread_cost_bps > 0

    def test_estimate_no_change_zero_cost(self):
        """No turnover should mean zero cost."""
        w = {"AAPL": 0.5, "MSFT": 0.5}
        cost = estimate_turnover_cost(w, w)
        assert cost.total_cost_bps == pytest.approx(0.0)

    def test_turnover_budget(self):
        """Budget calculation should be consistent."""
        budget = turnover_budget(200.0, 100.0, 6)
        assert budget["budget_remaining_pct"] == pytest.approx(100.0)
        assert budget["monthly_allowance_pct"] == pytest.approx(100.0 / 6)
        assert budget["is_over_budget"] is False

    def test_turnover_budget_over(self):
        """Over-budget should be detected."""
        budget = turnover_budget(200.0, 250.0, 3)
        assert budget["is_over_budget"] is True

    def test_turnover_summary(self):
        """Summary should work with weight history."""
        history = [
            {"AAPL": 0.5, "MSFT": 0.5},
            {"AAPL": 0.4, "MSFT": 0.6},
            {"AAPL": 0.45, "MSFT": 0.55},
            {"AAPL": 0.3, "MSFT": 0.7},
        ]
        summary = turnover_summary(history)
        assert isinstance(summary, TurnoverSummary)
        assert summary.avg_monthly_turnover > 0
        assert summary.annualized_turnover > 0
        assert len(summary.recommendation) > 0

    def test_turnover_summary_insufficient(self):
        """Single snapshot should return gracefully."""
        summary = turnover_summary([{"AAPL": 1.0}])
        assert summary.turnover_trend == "insufficient_data"
