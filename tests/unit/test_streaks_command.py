"""Tests for the `streaks` CLI command."""

import datetime as dt
import json

import polars as pl

from imst_quant.cli import cmd_streaks, create_parser


def _features(tmp_path, frame):
    path = tmp_path / "features.parquet"
    frame.write_parquet(path)
    return path


def test_streaks_requires_asset_for_multi_asset_features(tmp_path, capsys):
    path = _features(
        tmp_path,
        pl.DataFrame({"asset_id": ["X", "Y", "X", "Y"], "return_1d": [0.01, -0.01, 0.02, -0.02]}),
    )

    assert cmd_streaks(create_parser().parse_args(["streaks", "--features", str(path)])) == 1
    assert "specify --asset" in capsys.readouterr().out


def test_streaks_counts_runs_in_date_order(tmp_path, capsys):
    dates = [dt.date(2024, 1, d) for d in (4, 1, 3, 2)]
    path = _features(
        tmp_path,
        pl.DataFrame({"date": dates, "asset_id": ["X"] * 4, "return_1d": [-0.01, 0.01, -0.01, 0.01]}),
    )

    args = create_parser().parse_args(["streaks", "--features", str(path), "--asset", "X", "--json"])
    assert cmd_streaks(args) == 0

    payload = json.loads(capsys.readouterr().out)
    # Sorted by date: +, +, -, -  -> one 2-win and one 2-loss streak
    assert payload["max_win_streak"] == 2
    assert payload["max_loss_streak"] == 2
    assert payload["current_streak"] == -2
