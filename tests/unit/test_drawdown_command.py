"""Tests for drawdown duration stats and the `drawdown` CLI command."""

import datetime as dt

import polars as pl
import pytest

from imst_quant.cli import cmd_drawdown, create_parser
from imst_quant.utils.drawdown_analysis import drawdown_duration_analysis


def _returns_frame(returns):
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(returns))]
    return pl.DataFrame({"date": dates, "returns": returns})


def test_ongoing_drawdown_counts_toward_duration():
    """An unrecovered drawdown has no end index but is still the longest one."""
    returns = [0.02] + [-0.02] * 9

    stats = drawdown_duration_analysis(pl.Series("returns", returns))

    # Drawdown opens at index 1 and runs to the last bar (index 9): 8 periods,
    # measured the same way as a recovered drawdown's end_idx - start_idx.
    assert stats["max_drawdown_duration"] == 8
    assert stats["avg_drawdown_duration"] == pytest.approx(8.0)
    assert stats["pct_recovered"] == 0.0
    assert stats["avg_recovery_time"] == 0.0


def test_recovered_drawdown_duration_is_unchanged():
    stats = drawdown_duration_analysis(pl.Series("returns", [0.05, -0.10, 0.02, 0.05, 0.05, 0.02]))

    assert stats["max_drawdown_duration"] == 3
    assert stats["pct_recovered"] == pytest.approx(1.0)


def test_duration_analysis_accepts_a_dataframe():
    df = _returns_frame([0.05, -0.10, 0.02, 0.05, 0.05, 0.02])

    assert drawdown_duration_analysis(df) == drawdown_duration_analysis(df["returns"])


def test_flat_series_has_no_drawdowns():
    stats = drawdown_duration_analysis(pl.Series("returns", [0.01] * 5))

    assert stats["max_drawdown_duration"] == 0
    assert stats["pct_recovered"] == 0.0


def test_drawdown_command_is_registered():
    args = create_parser().parse_args(["drawdown", "--returns", "x.parquet", "--worst", "3"])

    assert args.command == "drawdown"
    assert args.worst == 3
    assert args.alpha == 0.95
    assert args.return_col == "returns"


def test_drawdown_command_reports_json(tmp_path, capsys):
    import json

    path = tmp_path / "returns.parquet"
    _returns_frame([0.02, -0.05, -0.03, 0.01, 0.06, 0.01, -0.02, 0.03] * 4).write_parquet(path)

    args = create_parser().parse_args(
        ["drawdown", "--returns", str(path), "--worst", "2", "--json"]
    )
    assert cmd_drawdown(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["statistics"]["max_drawdown"] > 0
    assert payload["drawdown_at_risk"]["levels"][0]["alpha"] == pytest.approx(0.95)
    assert len(payload["worst_drawdowns"]) <= 2
    assert payload["worst_drawdowns"][0]["trough_date"] is not None


def test_drawdown_command_rejects_bad_alpha(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _returns_frame([0.01, -0.02, 0.03]).write_parquet(path)

    args = create_parser().parse_args(["drawdown", "--returns", str(path), "--alpha", "1.5"])

    assert cmd_drawdown(args) == 1
    assert "--alpha" in capsys.readouterr().out


def test_drawdown_command_reports_missing_column(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    _returns_frame([0.01, -0.02, 0.03]).write_parquet(path)

    args = create_parser().parse_args(
        ["drawdown", "--returns", str(path), "--return-col", "pnl"]
    )

    assert cmd_drawdown(args) == 1
    assert "not found" in capsys.readouterr().out


def test_drawdown_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(["drawdown", "--returns", str(tmp_path / "nope.parquet")])

    assert cmd_drawdown(args) == 1
    assert "not found" in capsys.readouterr().out
