"""Tests for the `sentiment-strength` CLI command."""

import datetime as dt
import json

import polars as pl

from imst_quant.cli import cmd_sentiment_strength, create_parser


def _file(tmp_path):
    n = 20
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)]
    rows = []
    for name, base in (("BULL", 0.6), ("BEAR", -0.6), ("MIXED", 0.0)):
        for i, d in enumerate(dates):
            pol = base if name != "MIXED" else (0.5 if i % 2 else -0.5)
            rows.append({"date": d, "asset_id": name, "sentiment_index": pol, "post_count": 50})
    rows += [{"date": dates[0], "asset_id": "TINY", "sentiment_index": 0.9, "post_count": 50}]
    path = tmp_path / "sentiment.parquet"
    pl.DataFrame(rows).write_parquet(path)
    return path


def _run(path, capsys, *extra):
    args = create_parser().parse_args(
        ["sentiment-strength", "--data", str(path), "--json", *extra]
    )
    code = cmd_sentiment_strength(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_ranks_and_labels_direction(tmp_path, capsys):
    code, out = _run(_file(tmp_path), capsys)
    assert code == 0
    by_asset = {a["asset_id"]: a for a in out["assets"]}
    assert by_asset["BULL"]["direction"] == "bullish"
    assert by_asset["BEAR"]["direction"] == "bearish"
    # Steady one-sided sentiment beats flip-flopping sentiment
    assert by_asset["BULL"]["strength_score"] > by_asset["MIXED"]["strength_score"]
    assert by_asset["BEAR"]["strength_score"] > by_asset["MIXED"]["strength_score"]
    assert out["skipped_assets"] == ["TINY"]
    assert out["subjectivity_available"] is False


def test_top_limits_output(tmp_path, capsys):
    code, out = _run(_file(tmp_path), capsys, "--top", "1")
    assert code == 0 and len(out["assets"]) == 1 and out["n_assets"] == 3


def test_rejects_bad_input(tmp_path, capsys):
    path = _file(tmp_path)
    code, out = _run(path, capsys, "--top", "0")
    assert code == 1 and "--top" in out
    code, out = _run(path, capsys, "--polarity-col", "nope")
    assert code == 1 and "Missing columns" in out
    code, out = _run(tmp_path / "missing.parquet", capsys)
    assert code == 1 and "not found" in out
