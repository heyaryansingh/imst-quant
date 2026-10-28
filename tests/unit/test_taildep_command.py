"""Tests for the `taildep` CLI command."""

import datetime as dt
import json

import numpy as np
import polars as pl

from imst_quant.cli import cmd_taildep, create_parser


def _features(tmp_path, n=600):
    rng = np.random.default_rng(0)
    m = rng.normal(0, 0.01, n)
    # B crashes with A but rallies independently; C is independent of both
    a = m
    b = np.where(m < np.quantile(m, 0.1), m, rng.normal(0, 0.01, n))
    c = rng.normal(0, 0.01, n)
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rows = [
        {"date": d, "asset_id": k, "return_1d": float(v[i])}
        for i, d in enumerate(dates)
        for k, v in (("A", a), ("B", b), ("C", c))
    ]
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_taildep_ranks_crash_coupled_pair_first(tmp_path, capsys):
    path = _features(tmp_path)
    args = create_parser().parse_args(["taildep", "--features", str(path), "--json"])
    assert cmd_taildep(args) == 0

    out = json.loads(capsys.readouterr().out)
    assert out["n_assets"] == 3
    top = out["pairs"][0]
    assert {top["asset_a"], top["asset_b"]} == {"A", "B"}
    assert top["lower_tail_dep"] > 0.4  # independent pairs ~0.05
    assert top["crash_asymmetry"] > 0.3


def test_taildep_rejects_bad_quantile_and_short_history(tmp_path, capsys):
    path = _features(tmp_path, n=50)
    args = create_parser().parse_args(["taildep", "--features", str(path)])
    assert cmd_taildep(args) == 1  # 0.95 needs >= 100 dates
    args = create_parser().parse_args(
        ["taildep", "--features", str(path), "--quantile", "1.2"]
    )
    assert cmd_taildep(args) == 1


def test_taildep_text_output(tmp_path, capsys):
    path = _features(tmp_path)
    args = create_parser().parse_args(["taildep", "--features", str(path), "--top", "1"])
    assert cmd_taildep(args) == 0
    assert "Tail Dependence" in capsys.readouterr().out
