"""Tests for the `turnover` CLI command."""

import datetime as dt
import json

import polars as pl
import pytest

from imst_quant.cli import cmd_turnover, create_parser

# Alternates 0.5/0.5 and 0.6/0.4, so every step is 10% one-way turnover.
HISTORY = [
    {"AAPL": 0.5, "MSFT": 0.5},
    {"AAPL": 0.6, "MSFT": 0.4},
    {"AAPL": 0.5, "MSFT": 0.5},
    {"AAPL": 0.6, "MSFT": 0.4},
]


def _history_file(tmp_path, snapshots=HISTORY, dates=None, name="weight_history.parquet"):
    dates = dates or [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i)
                      for i in range(len(snapshots))]
    rows = [
        {"date": date, "symbol": symbol, "weight": weight}
        for date, snapshot in zip(dates, snapshots)
        for symbol, weight in snapshot.items()
    ]
    path = tmp_path / name
    pl.DataFrame(rows).write_parquet(path)
    return path


def test_turnover_command_is_registered():
    args = create_parser().parse_args(["turnover", "--history", "x.parquet"])

    assert args.command == "turnover"
    assert args.date_col == "date"
    assert args.months_elapsed == 0
    assert args.budget is None


def test_turnover_command_reports_json(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(["turnover", "--history", str(path), "--json"])
    assert cmd_turnover(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["snapshots"] == 4
    assert payload["avg_period_turnover"] == pytest.approx(0.10)
    assert payload["annualized_turnover"] == pytest.approx(1.2)
    assert payload["estimated_annual_cost_bps"] == pytest.approx(6.0)
    assert payload["first_date"] == "2024-01-01"
    assert payload["budget_remaining_pct"] is None


def test_turnover_command_sorts_snapshots_by_date(tmp_path, capsys):
    """Rows arriving out of order must not be differenced backwards in time."""
    ordered = _history_file(tmp_path, name="ordered.parquet")
    shuffled_dates = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(4)]
    scrambled = _history_file(
        tmp_path,
        snapshots=[HISTORY[2], HISTORY[0], HISTORY[3], HISTORY[1]],
        dates=[shuffled_dates[2], shuffled_dates[0], shuffled_dates[3], shuffled_dates[1]],
        name="scrambled.parquet",
    )

    args = create_parser().parse_args(["turnover", "--history", str(ordered), "--json"])
    assert cmd_turnover(args) == 0
    expected = json.loads(capsys.readouterr().out)

    args = create_parser().parse_args(["turnover", "--history", str(scrambled), "--json"])
    assert cmd_turnover(args) == 0
    actual = json.loads(capsys.readouterr().out)

    assert actual["avg_period_turnover"] == pytest.approx(expected["avg_period_turnover"])
    assert actual["first_date"] == expected["first_date"]
    assert actual["last_date"] == expected["last_date"]


def test_turnover_command_ranks_the_busiest_assets(tmp_path, capsys):
    path = _history_file(
        tmp_path,
        snapshots=[
            {"AAPL": 0.5, "MSFT": 0.3, "NVDA": 0.2},
            {"AAPL": 0.1, "MSFT": 0.3, "NVDA": 0.6},
        ],
    )

    args = create_parser().parse_args(["turnover", "--history", str(path), "--json"])
    assert cmd_turnover(args) == 0

    payload = json.loads(capsys.readouterr().out)
    busiest = payload["high_turnover_assets"]
    assert {busiest[0]["symbol"], busiest[1]["symbol"]} == {"AAPL", "NVDA"}
    assert busiest[-1]["symbol"] == "MSFT"
    assert busiest[-1]["turnover"] == pytest.approx(0.0)


def test_turnover_command_reports_budget_headroom(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(
        ["turnover", "--history", str(path), "--budget", "200",
         "--months-elapsed", "3", "--json"]
    )
    assert cmd_turnover(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["budget_remaining_pct"] == pytest.approx(200 - 30)


def test_turnover_command_prints_a_table(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(["turnover", "--history", str(path)])
    assert cmd_turnover(args) == 0

    out = capsys.readouterr().out
    assert "Portfolio Turnover" in out
    assert "Est. Cost Drag" in out
    assert "AAPL" in out


def test_turnover_command_needs_two_snapshots(tmp_path, capsys):
    path = _history_file(tmp_path, snapshots=[HISTORY[0]])

    args = create_parser().parse_args(["turnover", "--history", str(path)])

    assert cmd_turnover(args) == 1
    assert "at least 2 dated snapshots" in capsys.readouterr().out


def test_turnover_command_rejects_a_bad_budget(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(
        ["turnover", "--history", str(path), "--budget", "0"]
    )

    assert cmd_turnover(args) == 1
    assert "--budget" in capsys.readouterr().out


def test_turnover_command_rejects_bad_months_elapsed(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(
        ["turnover", "--history", str(path), "--months-elapsed", "13"]
    )

    assert cmd_turnover(args) == 1
    assert "--months-elapsed" in capsys.readouterr().out


def test_turnover_command_reports_missing_column(tmp_path, capsys):
    path = _history_file(tmp_path)

    args = create_parser().parse_args(
        ["turnover", "--history", str(path), "--weight-col", "target"]
    )

    assert cmd_turnover(args) == 1
    assert "not found" in capsys.readouterr().out


def test_turnover_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(
        ["turnover", "--history", str(tmp_path / "nope.parquet")]
    )

    assert cmd_turnover(args) == 1
    assert "not found" in capsys.readouterr().out
