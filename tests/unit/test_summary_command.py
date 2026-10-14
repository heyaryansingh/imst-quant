"""Tests for the `summary` CLI command."""

import json

import pandas as pd

from imst_quant.cli import cmd_summary, create_parser


def test_summary_date_range_uses_date_column(tmp_path, capsys):
    path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4),
            "ticker": list("AABB"),
            "returns": [0.01, -0.02, 0.03, 0.0],
        }
    ).to_parquet(path)
    args = create_parser().parse_args(["summary", "--features", str(path), "--json"])
    assert cmd_summary(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["date_range"]["start"].startswith("2024-01-01")
    assert payload["date_range"]["end"].startswith("2024-01-04")
    assert payload["top_tickers"] == {"A": 2, "B": 2}
