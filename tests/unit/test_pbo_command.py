"""Tests for probability of backtest overfitting and the `pbo` CLI command."""

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_pbo, create_parser
from imst_quant.utils.backtest_overfitting import probability_of_backtest_overfitting


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_noise_variants_have_coin_flip_pbo():
    noise = np.random.default_rng(0).normal(0, 0.01, (1000, 30))
    out = probability_of_backtest_overfitting(noise, n_blocks=10)
    assert out["n_splits"] == 252
    assert 0.3 < out["pbo"] < 0.7


def test_genuine_edge_has_low_pbo():
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, (1000, 10))
    r[:, 3] += 0.003  # one variant with a persistent edge
    out = probability_of_backtest_overfitting(r, n_blocks=10)
    assert out["pbo"] < 0.05
    assert out["prob_oos_loss"] < 0.05


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 3)), n_blocks=5)
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((10, 3)), n_blocks=16)


def test_cli_uses_numeric_columns_and_skips_nan_rows(tmp_path, capsys):
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, (400, 4))
    r[5, 1] = np.nan
    frame = pl.DataFrame({f"v{i}": r[:, i] for i in range(4)}).with_columns(
        pl.lit("x").alias("label")
    )
    path = tmp_path / "variants.parquet"
    frame.write_parquet(path)

    args = create_parser().parse_args(["pbo", "--data", str(path), "--blocks", "8", "--json"])
    assert cmd_pbo(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_strategies"] == 4
    assert payload["n_observations"] == 399

    args = create_parser().parse_args(["pbo", "--data", str(path), "--columns", "v0,nope"])
    assert cmd_pbo(args) == 1
