"""Tests for the `barriers` CLI command."""

import datetime as dt
import json

import numpy as np
import polars as pl

from imst_quant.cli import cmd_barriers, create_parser


def _file(tmp_path, tiny=False):
    rng = np.random.default_rng(1)
    n = 120
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rows = []
    for name, drift in (("UP", 0.01), ("FLAT", 0.0)):
        px = 100 * np.cumprod(1 + drift + rng.normal(0, 0.002, n))
        rows += [
            {"date": d, "asset_id": name, "close": float(p)}
            for d, p in zip(dates, px)
        ]
    if tiny:
        rows += [{"date": d, "asset_id": "TINY", "close": 1.0} for d in dates[:2]]
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _run(path, capsys, *extra):
    args = create_parser().parse_args(
        ["barriers", "--features", str(path), "--json", *extra]
    )
    code = cmd_barriers(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_barriers_registered():
    args = create_parser().parse_args(["barriers"])
    assert args.command == "barriers"
    assert args.max_holding == 10


def test_barriers_labels_trending_asset_upper(tmp_path, capsys):
    code, out = _run(_file(tmp_path, tiny=True), capsys)
    assert code == 0
    up = out["assets"]["UP"]
    assert up["proportions"]["upper"] > 0.8
    assert up["mean_return_by_label"]["upper"] > 0
    assert out["skipped_assets"] == ["TINY"]


def test_barriers_rejects_bad_input(tmp_path, capsys):
    path = _file(tmp_path)
    code, out = _run(path, capsys, "--pt-mult", "0")
    assert code == 1 and "--pt-mult" in out
    code, out = _run(path, capsys, "--max-holding", "0")
    assert code == 1 and "--max-holding" in out
    code, out = _run(tmp_path / "nope.parquet", capsys)
    assert code == 1 and "not found" in out
