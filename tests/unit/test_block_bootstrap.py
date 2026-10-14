"""Tests for block_bootstrap utilities and the `bootstrap` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_bootstrap, create_parser
from imst_quant.utils.block_bootstrap import bootstrap_metrics, stationary_bootstrap_indices


def test_indices_form_contiguous_blocks():
    idx = stationary_bootstrap_indices(50, 20, mean_block=10.0, rng=np.random.default_rng(0))

    assert idx.shape == (20, 50)
    assert idx.min() >= 0 and idx.max() < 50
    continued = (idx[:, 1:] == (idx[:, :-1] + 1) % 50).mean()
    assert continued == pytest.approx(0.9, abs=0.05)


def test_mean_block_one_is_iid():
    idx = stationary_bootstrap_indices(200, 50, mean_block=1.0, rng=np.random.default_rng(0))
    continued = (idx[:, 1:] == (idx[:, :-1] + 1) % 200).mean()
    assert continued < 0.02


def test_intervals_bracket_point_and_strong_edge_is_positive():
    r = np.random.default_rng(0).normal(0.003, 0.01, 750)

    out = bootstrap_metrics(r, n_resamples=300, seed=1)

    for name in ("sharpe", "annual_return", "max_drawdown"):
        assert out[name]["lower"] <= out[name]["point"] <= out[name]["upper"]
    assert out["sharpe"]["lower"] > 0
    assert out["prob_positive_sharpe"] == 1.0
    assert out["max_drawdown"]["upper"] <= 0


def test_drops_non_finite_and_is_seeded():
    r = list(np.random.default_rng(2).normal(0, 0.01, 40)) + [float("nan"), float("inf")]

    a = bootstrap_metrics(r, n_resamples=50, seed=3)
    assert a["n_observations"] == 40
    assert a == bootstrap_metrics(r, n_resamples=50, seed=3)


def test_total_loss_paths_clamp_annual_return():
    out = bootstrap_metrics([-1.0] + [0.01] * 19, n_resamples=50, seed=0)
    assert out["annual_return"]["point"] == -1.0
    assert out["max_drawdown"]["point"] == -1.0


@pytest.mark.parametrize(
    "kwargs", [{"confidence": 1.0}, {"n_resamples": 5}, {"periods_per_year": 0}]
)
def test_rejects_bad_parameters(kwargs):
    with pytest.raises(ValueError):
        bootstrap_metrics(np.zeros(20), **kwargs)


def test_cli_reports_json(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": np.random.default_rng(0).normal(0.001, 0.01, 100)}).write_parquet(path)

    args = create_parser().parse_args(
        ["bootstrap", "--data", str(path), "--resamples", "50", "--seed", "1", "--json"]
    )
    assert cmd_bootstrap(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_observations"] == 100
    assert payload["n_resamples"] == 50


def test_cli_rejects_short_series(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": [0.01, 0.02]}).write_parquet(path)

    assert cmd_bootstrap(create_parser().parse_args(["bootstrap", "--data", str(path)])) == 1
    assert "at least 10" in capsys.readouterr().out
