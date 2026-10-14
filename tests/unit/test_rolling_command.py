"""Tests for the `rolling` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_rolling, create_parser


def _run(tmp_path, capsys, returns, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": returns}).write_parquet(path)
    args = create_parser().parse_args(["rolling", "--data", str(path), *extra])
    code = cmd_rolling(args)
    return code, capsys.readouterr().out


def test_rolling_json(tmp_path, capsys):
    rng = np.random.default_rng(0)
    returns = rng.normal(0.001, 0.01, 200)
    code, out = _run(tmp_path, capsys, returns, "--window", "20", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["n_observations"] == 200
    sharpe = payload["metrics"]["sharpe"]
    assert sharpe["min"] <= sharpe["median"] <= sharpe["max"]
    assert 0.0 <= payload["pct_windows_positive_sharpe"] <= 1.0
    assert payload["current_drawdown"] <= 0.0


def test_always_winning_strategy(tmp_path, capsys):
    returns = np.linspace(0.001, 0.002, 50)
    code, out = _run(tmp_path, capsys, returns, "--window", "10", "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["pct_windows_positive_sharpe"] == 1.0
    assert payload["metrics"]["win_rate"]["latest"] == 1.0
    assert payload["current_drawdown"] == 0.0


def test_text_output(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [0.01, -0.01] * 30, "--window", "10")
    assert code == 0
    assert "Rolling Performance" in out


@pytest.mark.parametrize(
    "extra", [["--returns-col", "nope"], ["--window", "1"], ["--window", "500"]]
)
def test_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, [0.01, -0.01] * 30, *extra)
    assert code == 1
    assert "Error" in out


def test_nan_does_not_stale_latest_window(tmp_path, capsys):
    clean = list(np.random.default_rng(0).normal(0.001, 0.01, 100))
    _, out = _run(tmp_path, capsys, clean, "--window", "20", "--json")
    expected = json.loads(out)

    dirty = clean[:97] + [float("nan")] + clean[97:]
    _, out = _run(tmp_path, capsys, dirty, "--window", "20", "--json")
    payload = json.loads(out)

    assert payload["n_observations"] == 100
    assert payload["metrics"]["sharpe"]["latest"] == pytest.approx(
        expected["metrics"]["sharpe"]["latest"]
    )
