"""Tests for the `kelly` CLI command."""

import datetime as dt
import json

import polars as pl
import pytest

from imst_quant.cli import cmd_kelly, create_parser

WINNERS_AND_LOSERS = [0.02, -0.01, 0.03, -0.01, 0.04, -0.02, 0.02, -0.01]


def _returns_file(tmp_path, returns, name="returns.parquet"):
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(returns))]
    path = tmp_path / name
    pl.DataFrame({"date": dates, "returns": returns}).write_parquet(path)
    return path


def test_kelly_command_is_registered():
    args = create_parser().parse_args(["kelly", "--returns", "x.parquet"])

    assert args.command == "kelly"
    assert args.fraction == 0.25
    assert args.return_col == "returns"
    assert args.capital is None


def test_kelly_command_reports_json(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(["kelly", "--returns", str(path), "--json"])
    assert cmd_kelly(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_periods"] == len(WINNERS_AND_LOSERS)
    assert payload["win_rate"] == pytest.approx(0.5)
    assert payload["recommended_size"] == pytest.approx(payload["full_kelly"] * 0.25)
    assert "recommended_notional" not in payload


def test_kelly_command_scales_by_capital(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(
        ["kelly", "--returns", str(path), "--fraction", "0.5", "--capital", "50000", "--json"]
    )
    assert cmd_kelly(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["recommended_size"] == pytest.approx(payload["kelly_0.5"])
    assert payload["recommended_notional"] == pytest.approx(payload["recommended_size"] * 50000)


def test_kelly_command_recommends_nothing_without_an_edge(tmp_path, capsys):
    path = _returns_file(tmp_path, [-0.01, -0.02, -0.005, -0.03])

    args = create_parser().parse_args(["kelly", "--returns", str(path)])
    assert cmd_kelly(args) == 0

    out = capsys.readouterr().out
    assert "no position" in out


def test_kelly_command_prints_a_table(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(["kelly", "--returns", str(path)])
    assert cmd_kelly(args) == 0

    out = capsys.readouterr().out
    assert "Kelly Position Sizing" in out
    assert "<- recommended" in out


def test_kelly_command_rejects_a_bad_fraction(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(["kelly", "--returns", str(path), "--fraction", "1.5"])

    assert cmd_kelly(args) == 1
    assert "--fraction" in capsys.readouterr().out


def test_kelly_command_rejects_non_positive_capital(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(["kelly", "--returns", str(path), "--capital", "0"])

    assert cmd_kelly(args) == 1
    assert "--capital" in capsys.readouterr().out


def test_kelly_command_reports_missing_column(tmp_path, capsys):
    path = _returns_file(tmp_path, WINNERS_AND_LOSERS)

    args = create_parser().parse_args(["kelly", "--returns", str(path), "--return-col", "pnl"])

    assert cmd_kelly(args) == 1
    assert "not found" in capsys.readouterr().out


def test_kelly_command_reports_an_all_null_column(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": [None, None]}, schema={"returns": pl.Float64}).write_parquet(path)

    args = create_parser().parse_args(["kelly", "--returns", str(path)])

    assert cmd_kelly(args) == 1
    assert "no non-null values" in capsys.readouterr().out


def test_kelly_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(["kelly", "--returns", str(tmp_path / "nope.parquet")])

    assert cmd_kelly(args) == 1
    assert "not found" in capsys.readouterr().out
