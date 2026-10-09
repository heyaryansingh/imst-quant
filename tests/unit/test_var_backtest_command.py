"""Tests for out-of-sample VaR backtesting and the `var-backtest` command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_var_backtest, create_parser
from imst_quant.utils.var_backtesting import rolling_historical_var, var_backtest_summary


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _regime_shift(seed=0):
    rng = np.random.default_rng(seed)
    return np.concatenate([rng.normal(0, 0.005, 600), rng.normal(0, 0.03, 300)])


def test_rolling_var_has_no_lookahead():
    r = np.random.default_rng(1).normal(0, 0.01, 300)
    base = rolling_historical_var(r, window=100)
    shocked = r.copy()
    shocked[200] = -0.5
    after = rolling_historical_var(shocked, window=100)
    assert np.all(np.isnan(base[:100]))
    assert base[200] == after[200]  # bar 200's own return can't move its forecast
    assert after[201] >= base[201]


def test_summary_drops_warmup_nans():
    r = np.random.default_rng(2).normal(0, 0.01, 400)
    s = var_backtest_summary(r, rolling_historical_var(r, window=100))
    assert s["num_observations"] == 300


def test_cli_flags_volatility_regime_shift(tmp_path, capsys):
    # A full-sample constant VaR hits ~5% by construction and calls this
    # adequate; the out-of-sample forecast lags the jump and gets rejected.
    r = _regime_shift()
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": r}).write_parquet(path)
    args = create_parser().parse_args(
        ["var-backtest", "--returns", str(path), "--window", "250", "--json"]
    )
    assert cmd_var_backtest(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["num_observations"] == len(r) - 250
    assert payload["violation_rate"] > 0.07
    assert payload["model_adequate"] is False
