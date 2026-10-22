"""Tests for the `sectors` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_sectors, create_parser


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "sentiment.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["sectors", "--sentiment", str(path), *extra])
    code = cmd_sectors(args)
    return code, capsys.readouterr().out


def _df():
    return pl.DataFrame({
        "date": ["2026-10-01", "2026-10-02"] * 4,
        "asset_id": ["aapl", "aapl", "MSFT", "MSFT", "JPM", "JPM", "XOM", "XOM"],
        # Older rows are deliberately opposite so using them would flip sectors.
        "sentiment_index": [-0.9, 0.8, -0.9, float("nan"), 0.9, -0.5, 0.0, 0.0],
    })


def test_sectors_uses_latest_finite_row_per_asset(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _df(), "--json")
    assert code == 0
    payload = json.loads(out)
    metrics = {s["sector"]: s for s in payload["sector_metrics"]}
    # AAPL latest = 0.8; MSFT latest is NaN so its last finite row (-0.9) is used.
    assert metrics["Technology"]["avg_sentiment"] == pytest.approx(-0.05)
    assert metrics["Financials"]["avg_sentiment"] == pytest.approx(-0.5)
    assert payload["rotation_signal"]["underweight_sectors"] == ["Financials"]


def test_sectors_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, _df())
    assert code == 0
    assert "Underweight: Financials" in out


def test_sectors_rejects_unmapped_or_missing(tmp_path, capsys):
    df = pl.DataFrame({"asset_id": ["ZZZZ"], "sentiment_index": [0.5]})
    code, out = _run(tmp_path, capsys, df)
    assert code == 1
    code, out = _run(tmp_path, capsys, df, "--sentiment-col", "nope")
    assert code == 1
    assert "nope" in out
