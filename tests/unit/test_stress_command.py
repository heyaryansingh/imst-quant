"""Tests for the `stress` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_stress, create_parser


def _portfolio_file(tmp_path, symbols, weights, name="portfolio.parquet"):
    path = tmp_path / name
    pl.DataFrame({"symbol": symbols, "weight": weights}).write_parquet(path)
    return path


def test_stress_command_is_registered():
    args = create_parser().parse_args(["stress", "--portfolio", "x.parquet"])

    assert args.command == "stress"
    assert args.symbol_col == "symbol"
    assert args.weight_col == "weight"
    assert args.scenario is None
    assert args.capital is None


def test_list_scenarios_needs_no_portfolio(capsys):
    args = create_parser().parse_args(["stress", "--list-scenarios"])
    assert cmd_stress(args) == 0

    assert "2008_financial_crisis" in capsys.readouterr().out


def test_reports_impact_per_scenario(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["SPY", "TLT"], [0.6, 0.4])

    args = create_parser().parse_args(
        ["stress", "--portfolio", str(path), "--scenario", "2020_covid_crash", "--json"]
    )
    assert cmd_stress(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_positions"] == 2
    assert payload["worst_case"] == "2020_covid_crash"
    assert len(payload["scenarios"]) == 1

    # SPY -34% at 60% weight, TLT +15% at 40% weight.
    scenario = payload["scenarios"][0]
    assert scenario["portfolio_impact"] == pytest.approx(0.6 * -0.34 + 0.4 * 0.15)
    assert scenario["worst_asset"] == "SPY"
    assert scenario["best_asset"] == "TLT"
    assert "pnl" not in scenario


def test_capital_translates_impact_into_dollars(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["SPY"], [1.0])

    args = create_parser().parse_args([
        "stress", "--portfolio", str(path),
        "--scenario", "2020_covid_crash", "--capital", "50000", "--json",
    ])
    assert cmd_stress(args) == 0

    scenario = json.loads(capsys.readouterr().out)["scenarios"][0]
    assert scenario["pnl"] == pytest.approx(-0.34 * 50000)


def test_coverage_flags_positions_the_scenario_says_nothing_about(tmp_path, capsys):
    # NVDA is absent from every historical scenario, so it is shocked by zero
    # and the headline impact understates the real risk.
    path = _portfolio_file(tmp_path, ["SPY", "NVDA"], [0.2, 0.8])

    args = create_parser().parse_args(
        ["stress", "--portfolio", str(path), "--scenario", "2020_covid_crash", "--json"]
    )
    assert cmd_stress(args) == 0

    scenario = json.loads(capsys.readouterr().out)["scenarios"][0]
    assert scenario["weight_covered"] == pytest.approx(0.2)
    assert scenario["asset_contributions"]["NVDA"] == 0.0


def test_undercovered_scenarios_are_called_out_in_text_output(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["SPY", "NVDA"], [0.2, 0.8])

    args = create_parser().parse_args(
        ["stress", "--portfolio", str(path), "--scenario", "2020_covid_crash"]
    )
    assert cmd_stress(args) == 0

    assert "understated" in capsys.readouterr().out


def test_missing_portfolio_file_exits_nonzero(tmp_path, capsys):
    args = create_parser().parse_args(
        ["stress", "--portfolio", str(tmp_path / "absent.parquet")]
    )
    assert cmd_stress(args) == 1
    assert "not found" in capsys.readouterr().out


def test_unknown_scenario_exits_nonzero(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["SPY"], [1.0])

    args = create_parser().parse_args(
        ["stress", "--portfolio", str(path), "--scenario", "y2k_bug"]
    )
    assert cmd_stress(args) == 1
    assert "Unknown scenario" in capsys.readouterr().out


def test_missing_weight_column_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "portfolio.parquet"
    pl.DataFrame({"symbol": ["SPY"], "allocation": [1.0]}).write_parquet(path)

    args = create_parser().parse_args(["stress", "--portfolio", str(path)])
    assert cmd_stress(args) == 1
    assert "'weight' not found" in capsys.readouterr().out


def test_empty_portfolio_exits_nonzero(tmp_path, capsys):
    path = tmp_path / "portfolio.parquet"
    pl.DataFrame(
        {"symbol": [None], "weight": [None]},
        schema={"symbol": pl.String, "weight": pl.Float64},
    ).write_parquet(path)

    args = create_parser().parse_args(["stress", "--portfolio", str(path)])
    assert cmd_stress(args) == 1
    assert "no rows" in capsys.readouterr().out


def test_non_positive_capital_exits_nonzero(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["SPY"], [1.0])

    args = create_parser().parse_args(
        ["stress", "--portfolio", str(path), "--capital", "0"]
    )
    assert cmd_stress(args) == 1
    assert "must be positive" in capsys.readouterr().out
