"""Tests for the `tail-risk` CLI command."""

import datetime as dt
import json

import polars as pl
import pytest

from imst_quant.cli import cmd_tail_risk, create_parser

MIXED_RETURNS = [0.01, -0.02, 0.015, -0.05, 0.02, -0.01, 0.03, -0.04]


def _returns_file(tmp_path, returns, name="returns.parquet"):
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(len(returns))]
    path = tmp_path / name
    pl.DataFrame({"date": dates, "returns": returns}).write_parquet(path)
    return path


def test_tail_risk_command_is_registered():
    args = create_parser().parse_args(["tail-risk", "--returns", "x.parquet"])

    assert args.command == "tail-risk"
    assert args.confidence == 0.95
    assert args.return_col == "returns"
    assert args.capital is None


def test_tail_risk_command_reports_json(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(["tail-risk", "--returns", str(path), "--json"])
    assert cmd_tail_risk(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_periods"] == len(MIXED_RETURNS)
    assert payload["confidence"] == 0.95
    assert payload["cvar"] > 0
    assert set(payload) >= {"cvar", "tail_ratio", "omega_ratio", "evt_var"}
    assert "cvar_notional" not in payload


def test_tail_risk_command_scales_by_capital(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(
        ["tail-risk", "--returns", str(path), "--capital", "50000", "--json"]
    )
    assert cmd_tail_risk(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["cvar_notional"] == pytest.approx(payload["cvar"] * 50000)
    assert payload["evt_var_notional"] == pytest.approx(payload["evt_var"] * 50000)


def test_tail_risk_command_ignores_nulls(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS + [None])

    args = create_parser().parse_args(["tail-risk", "--returns", str(path), "--json"])
    assert cmd_tail_risk(args) == 0

    assert json.loads(capsys.readouterr().out)["n_periods"] == len(MIXED_RETURNS)


def test_tail_risk_command_prints_a_table(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(["tail-risk", "--returns", str(path)])
    assert cmd_tail_risk(args) == 0

    out = capsys.readouterr().out
    assert "Tail Risk" in out
    assert "CVaR" in out
    assert "Omega Ratio" in out


def test_tail_risk_command_warns_when_evt_falls_back(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(["tail-risk", "--returns", str(path)])
    assert cmd_tail_risk(args) == 0

    assert "empirical quantile" in capsys.readouterr().out


def test_tail_risk_command_rejects_a_bad_confidence(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(
        ["tail-risk", "--returns", str(path), "--confidence", "1.0"]
    )

    assert cmd_tail_risk(args) == 1
    assert "--confidence" in capsys.readouterr().out


def test_tail_risk_command_rejects_non_positive_capital(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(
        ["tail-risk", "--returns", str(path), "--capital", "0"]
    )

    assert cmd_tail_risk(args) == 1
    assert "--capital" in capsys.readouterr().out


def test_tail_risk_command_reports_missing_column(tmp_path, capsys):
    path = _returns_file(tmp_path, MIXED_RETURNS)

    args = create_parser().parse_args(
        ["tail-risk", "--returns", str(path), "--return-col", "pnl"]
    )

    assert cmd_tail_risk(args) == 1
    assert "not found" in capsys.readouterr().out


def test_tail_risk_command_reports_an_all_null_column(tmp_path, capsys):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": [None, None]}, schema={"returns": pl.Float64}).write_parquet(path)

    args = create_parser().parse_args(["tail-risk", "--returns", str(path)])

    assert cmd_tail_risk(args) == 1
    assert "no non-null values" in capsys.readouterr().out


def test_tail_risk_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(
        ["tail-risk", "--returns", str(tmp_path / "nope.parquet")]
    )

    assert cmd_tail_risk(args) == 1
    assert "not found" in capsys.readouterr().out
