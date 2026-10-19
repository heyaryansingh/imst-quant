"""Tests for the `signal` CLI command."""

import json
import sys

import pandas as pd
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_signal, create_parser
from imst_quant.utils.signal_backtest import backtest_signal


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, frame, *extra):
    path = tmp_path / "features.parquet"
    pl.DataFrame(frame).write_parquet(path)
    args = create_parser().parse_args(
        ["signal", "--features", str(path), "--signal-col", "signal", *extra]
    )
    code = cmd_signal(args)
    return code, capsys.readouterr().out


def test_signal_drops_incomplete_rows_jointly(tmp_path, capsys):
    signal = [1.0, None, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0]
    returns = [0.01, 0.02, -0.01, float("nan"), 0.03, -0.02, 0.01, 0.02, -0.01, 0.01]
    code, out = _run(tmp_path, capsys, {"signal": signal, "returns": returns}, "--json")
    assert code == 0

    keep = [i for i in range(len(signal)) if signal[i] is not None and returns[i] == returns[i]]
    expected = backtest_signal(
        pd.Series([signal[i] for i in keep]),
        pd.Series([returns[i] for i in keep]),
        transaction_cost=0.001,
    )
    got = json.loads(out)["backtest"]
    assert got["total_return"] == pytest.approx(expected.total_return)
    assert got["n_trades"] == expected.n_trades


def test_signal_rejects_too_few_complete_rows(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, {"signal": [1.0, None], "returns": [None, 0.01]})
    assert code == 1
    assert "Error" in out
