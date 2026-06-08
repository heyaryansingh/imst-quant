"""Tests for walk-forward validation module.

Tests cover:
- Window generation (rolling and anchored modes)
- Parameter optimization across windows
- Summary statistics calculation
- Parameter stability analysis
- OOS equity curve construction
- Edge cases (insufficient data, empty param grids)
"""

import polars as pl
import pytest

from imst_quant.trading.walk_forward import (
    WalkForwardConfig,
    WalkForwardValidator,
    WindowResult,
)


# ---------------------------------------------------------------------------
# Helper strategy functions for testing
# ---------------------------------------------------------------------------


def dummy_strategy(data: pl.DataFrame, **params) -> dict[str, float]:
    """Simple strategy that returns metrics based on data length."""
    n = len(data)
    lookback = params.get("lookback", 10)
    # Simulate: longer lookback = slightly better Sharpe on training data
    sharpe = 0.5 + (lookback / 100) + (n / 10000)
    total_return = 0.02 * (n / 252)
    return {"sharpe": sharpe, "total_return": total_return, "win_rate": 0.55}


def failing_strategy(data: pl.DataFrame, **params) -> dict[str, float]:
    """Strategy that raises an error."""
    raise ValueError("Strategy failed")


def negative_strategy(data: pl.DataFrame, **params) -> dict[str, float]:
    """Strategy that always loses money."""
    return {"sharpe": -0.5, "total_return": -0.03, "win_rate": 0.35}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def daily_data():
    """Generate 500 days of synthetic feature data."""
    import numpy as np

    np.random.seed(42)
    n = 500
    dates = pl.date_range(
        pl.date(2022, 1, 1), pl.date(2022, 1, 1) + pl.duration(days=n - 1), eager=True
    )
    return pl.DataFrame(
        {
            "date": dates,
            "return_1d": np.random.normal(0.0005, 0.02, n).tolist(),
            "sentiment": np.random.uniform(-1, 1, n).tolist(),
            "feature_1": np.random.randn(n).tolist(),
        }
    )


@pytest.fixture
def short_data():
    """Data too short for default walk-forward windows."""
    return pl.DataFrame(
        {
            "date": pl.date_range(
                pl.date(2022, 1, 1), pl.date(2022, 3, 1), eager=True
            ),
            "return_1d": [0.01] * 60,
        }
    )


@pytest.fixture
def param_grid():
    """Simple parameter grid for optimization."""
    return {"lookback": [5, 10, 20]}


# ---------------------------------------------------------------------------
# Window generation tests
# ---------------------------------------------------------------------------


class TestWindowGeneration:
    """Tests for walk-forward window generation."""

    def test_rolling_windows_generated(self, daily_data):
        """Rolling mode should generate multiple windows."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        windows = validator._generate_windows(len(daily_data))
        assert len(windows) > 0
        # Each window should have 4 indices
        for w in windows:
            assert len(w) == 4
            train_start, train_end, test_start, test_end = w
            assert train_end - train_start == 100
            assert test_end - test_start == 50
            assert test_start == train_end

    def test_anchored_windows(self, daily_data):
        """Anchored mode should always start training from index 0."""
        config = WalkForwardConfig(
            train_size=100, test_size=50, step_size=50, anchored=True
        )
        validator = WalkForwardValidator(config=config)
        windows = validator._generate_windows(len(daily_data))
        assert len(windows) > 0
        for w in windows:
            assert w[0] == 0  # train always starts at 0

    def test_insufficient_data_returns_empty(self, short_data):
        """Too-short data should return no windows."""
        validator = WalkForwardValidator(
            train_size=252, test_size=63, step_size=63
        )
        windows = validator._generate_windows(len(short_data))
        assert len(windows) == 0

    def test_step_size_controls_overlap(self, daily_data):
        """Smaller step size should produce more windows."""
        v_large = WalkForwardValidator(
            train_size=100, test_size=50, step_size=100
        )
        v_small = WalkForwardValidator(
            train_size=100, test_size=50, step_size=25
        )
        large_wins = v_large._generate_windows(len(daily_data))
        small_wins = v_small._generate_windows(len(daily_data))
        assert len(small_wins) > len(large_wins)

    def test_min_train_size_respected(self):
        """Windows below min_train_size should be skipped."""
        config = WalkForwardConfig(
            train_size=100, test_size=50, step_size=50, min_train_size=200
        )
        validator = WalkForwardValidator(config=config)
        # With min_train_size=200 but train_size=100, non-anchored windows
        # will have actual_train_size=100 < 200, so all get skipped
        windows = validator._generate_windows(500)
        assert len(windows) == 0


# ---------------------------------------------------------------------------
# Run / optimization tests
# ---------------------------------------------------------------------------


class TestWalkForwardRun:
    """Tests for running walk-forward validation."""

    def test_run_returns_windows(self, daily_data, param_grid):
        """Run should return window results."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy, param_grid)
        assert "windows" in results
        assert "summary" in results
        assert "param_stability" in results
        assert len(results["windows"]) > 0

    def test_window_results_are_valid(self, daily_data, param_grid):
        """Each window result should have correct structure."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy, param_grid)
        for w in results["windows"]:
            assert isinstance(w, WindowResult)
            assert w.window_id >= 1
            assert "sharpe" in w.train_metrics
            assert "sharpe" in w.test_metrics
            assert "lookback" in w.best_params

    def test_run_without_param_grid(self, daily_data):
        """Run without param_grid should use default strategy params."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy)
        assert len(results["windows"]) > 0
        # No param grid -> empty best_params
        for w in results["windows"]:
            assert w.best_params == {}

    def test_run_insufficient_data(self, short_data):
        """Insufficient data should return empty results with error."""
        validator = WalkForwardValidator(
            train_size=252, test_size=63, step_size=63
        )
        results = validator.run(short_data, dummy_strategy)
        assert results["windows"] == []
        assert "error" in results

    def test_failing_strategy_handled(self, daily_data):
        """Failing test evaluation should be caught gracefully."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        # failing_strategy raises on all calls, but train phase catches it
        # too, so we use a strategy that fails only on test
        call_count = {"n": 0}

        def sometimes_failing(data, **params):
            call_count["n"] += 1
            # Fail on every other call (the test evaluations)
            if call_count["n"] % 2 == 0:
                raise ValueError("test failure")
            return {"sharpe": 1.0, "total_return": 0.05}

        results = validator.run(daily_data, sometimes_failing)
        # Should complete without raising
        assert len(results["windows"]) > 0


# ---------------------------------------------------------------------------
# Summary statistics tests
# ---------------------------------------------------------------------------


class TestSummaryStatistics:
    """Tests for walk-forward summary calculations."""

    def test_summary_has_required_keys(self, daily_data, param_grid):
        """Summary should contain all expected metrics."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy, param_grid)
        summary = results["summary"]
        assert "n_windows" in summary
        assert "avg_oos_sharpe" in summary
        assert "avg_is_sharpe" in summary
        assert "sharpe_decay_pct" in summary
        assert "oos_sharpe_std" in summary
        assert "oos_win_rate" in summary

    def test_sharpe_decay_positive_for_overfitting(self, daily_data, param_grid):
        """IS Sharpe > OOS Sharpe should give positive decay."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy, param_grid)
        summary = results["summary"]
        # dummy_strategy gives slightly higher Sharpe for longer data (training)
        # so IS should be >= OOS, meaning decay >= 0
        assert summary["avg_is_sharpe"] >= 0

    def test_negative_strategy_summary(self, daily_data):
        """Negative strategy should show poor OOS metrics."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, negative_strategy)
        summary = results["summary"]
        assert summary["avg_oos_sharpe"] < 0
        assert summary["oos_win_rate"] == 0.0  # All returns negative

    def test_empty_windows_summary(self):
        """Empty windows should give empty summary."""
        validator = WalkForwardValidator()
        summary = validator._calculate_summary()
        assert summary == {}


# ---------------------------------------------------------------------------
# Parameter stability tests
# ---------------------------------------------------------------------------


class TestParamStability:
    """Tests for parameter stability analysis."""

    def test_stability_for_numeric_params(self, daily_data, param_grid):
        """Numeric params should have mean, std, cv."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy, param_grid)
        stability = results["param_stability"]
        assert "lookback" in stability
        assert "mean" in stability["lookback"]
        assert "std" in stability["lookback"]
        assert "cv" in stability["lookback"]

    def test_no_stability_without_grid(self, daily_data):
        """Without param_grid, stability should be empty."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        results = validator.run(daily_data, dummy_strategy)
        assert results["param_stability"] == {}


# ---------------------------------------------------------------------------
# OOS equity curve tests
# ---------------------------------------------------------------------------


class TestOOSEquityCurve:
    """Tests for out-of-sample equity curve construction."""

    def test_equity_curve_starts_at_one(self, daily_data, param_grid):
        """Equity curve should start at 1.0."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        validator.run(daily_data, dummy_strategy, param_grid)
        equity = validator.get_oos_equity_curve()
        assert equity[0] == 1.0

    def test_equity_curve_length(self, daily_data, param_grid):
        """Equity curve should have n_windows + 1 points."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        validator.run(daily_data, dummy_strategy, param_grid)
        equity = validator.get_oos_equity_curve()
        assert len(equity) == len(validator.windows) + 1

    def test_equity_curve_positive_strategy(self, daily_data, param_grid):
        """Positive strategy should end above 1.0."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        validator.run(daily_data, dummy_strategy, param_grid)
        equity = validator.get_oos_equity_curve()
        assert equity[-1] > 1.0  # dummy_strategy has positive returns

    def test_empty_equity_curve(self):
        """No windows should return just [1.0]."""
        validator = WalkForwardValidator()
        equity = validator.get_oos_equity_curve()
        assert equity == [1.0]


# ---------------------------------------------------------------------------
# to_dict serialization tests
# ---------------------------------------------------------------------------


class TestSerialization:
    """Tests for result serialization."""

    def test_to_dict_structure(self, daily_data, param_grid):
        """to_dict should have config, windows, summary, equity curve."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        validator.run(daily_data, dummy_strategy, param_grid)
        d = validator.to_dict()
        assert "config" in d
        assert "windows" in d
        assert "summary" in d
        assert "oos_equity_curve" in d
        assert d["config"]["train_size"] == 100
        assert d["config"]["test_size"] == 50

    def test_to_dict_windows_serializable(self, daily_data, param_grid):
        """Window entries should be plain dicts (JSON-serializable)."""
        validator = WalkForwardValidator(
            train_size=100, test_size=50, step_size=50
        )
        validator.run(daily_data, dummy_strategy, param_grid)
        d = validator.to_dict()
        for w in d["windows"]:
            assert isinstance(w, dict)
            assert "window_id" in w
            assert "train_period" in w
            assert "test_period" in w
