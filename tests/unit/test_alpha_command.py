"""Tests for the `alpha` CLI command."""

import argparse
import datetime as dt
import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_alpha, create_parser


def _write_returns(path, n=250, with_benchmark=True, seed=0):
    rng = np.random.default_rng(seed)
    benchmark = rng.normal(0.0004, 0.01, n)
    strategy = 1.2 * benchmark + rng.normal(0.0003, 0.004, n)
    data = {
        "date": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "returns": strategy,
    }
    if with_benchmark:
        data["benchmark_returns"] = benchmark
    pl.DataFrame(data).write_parquet(path)
    return benchmark


def _args(**overrides):
    defaults = dict(
        returns=None,
        return_col="returns",
        benchmark_col="benchmark_returns",
        benchmark=None,
        risk_free_rate=0.02,
        simulations=25,
        seed=1,
        json=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_alpha_is_registered_with_the_parser():
    args = create_parser().parse_args(["alpha", "--json", "--seed", "4"])
    assert args.command == "alpha"
    assert args.json is True
    assert args.seed == 4
    assert args.risk_free_rate == pytest.approx(0.02)


def test_reports_metrics_from_a_single_file(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _write_returns(path)

    assert cmd_alpha(_args(returns=str(path))) == 0

    out = capsys.readouterr().out
    assert "Alpha Metrics" in out
    assert "Information Ratio" in out
    assert "Prob. Sharpe Ratio" in out


def test_json_output_carries_every_metric(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _write_returns(path)

    assert cmd_alpha(_args(returns=str(path), json=True)) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["observations"] == 250
    # Strategy is built as 1.2x the benchmark plus noise.
    assert payload["beta"] == pytest.approx(1.2, abs=0.15)
    assert 0.0 <= payload["probabilistic_sharpe_ratio"] <= 1.0


def test_benchmark_can_live_in_a_separate_file(tmp_path, capsys):
    returns_path = tmp_path / "returns.parquet"
    bench_path = tmp_path / "bench.parquet"
    benchmark = _write_returns(returns_path, with_benchmark=False)
    pl.DataFrame({"benchmark_returns": benchmark}).write_parquet(bench_path)

    exit_code = cmd_alpha(
        _args(returns=str(returns_path), benchmark=str(bench_path), json=True)
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out)["observations"] == 250


def test_missing_returns_file_is_reported(tmp_path, capsys):
    assert cmd_alpha(_args(returns=str(tmp_path / "nope.parquet"))) == 1
    assert "not found" in capsys.readouterr().out


def test_missing_benchmark_column_is_reported(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _write_returns(path, with_benchmark=False)

    assert cmd_alpha(_args(returns=str(path))) == 1
    assert "benchmark_returns" in capsys.readouterr().out


def test_mismatched_lengths_are_rejected(tmp_path, capsys):
    returns_path = tmp_path / "returns.parquet"
    bench_path = tmp_path / "bench.parquet"
    _write_returns(returns_path, n=250, with_benchmark=False)
    pl.DataFrame({"benchmark_returns": np.zeros(10)}).write_parquet(bench_path)

    exit_code = cmd_alpha(
        _args(returns=str(returns_path), benchmark=str(bench_path))
    )

    assert exit_code == 1
    assert "aligned" in capsys.readouterr().out


def test_single_observation_is_rejected(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": [0.01], "benchmark_returns": [0.02]}).write_parquet(path)

    assert cmd_alpha(_args(returns=str(path))) == 1
    assert "at least 2 observations" in capsys.readouterr().out


def test_seed_makes_output_reproducible(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _write_returns(path)

    cmd_alpha(_args(returns=str(path), json=True, seed=9))
    first = capsys.readouterr().out
    cmd_alpha(_args(returns=str(path), json=True, seed=9))
    second = capsys.readouterr().out

    assert first == second
