"""Tests for the `imst attribution` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import COMMANDS, cmd_attribution, create_parser


@pytest.fixture
def holdings(tmp_path):
    """Write a portfolio/benchmark pair and return their paths."""
    portfolio = pl.DataFrame({
        "sector": ["Tech", "Energy"],
        "weight": [0.7, 0.3],
        "return": [0.12, 0.02],
    })
    benchmark = pl.DataFrame({
        "sector": ["Tech", "Energy"],
        "weight": [0.5, 0.5],
        "return": [0.10, 0.03],
    })
    portfolio_path = tmp_path / "portfolio.parquet"
    benchmark_path = tmp_path / "benchmark.parquet"
    portfolio.write_parquet(portfolio_path)
    benchmark.write_parquet(benchmark_path)
    return portfolio_path, benchmark_path


def _args(portfolio, benchmark, **overrides):
    parser = create_parser()
    argv = ["attribution", "--portfolio", str(portfolio), "--benchmark", str(benchmark)]
    for key, value in overrides.items():
        argv.extend([f"--{key.replace('_', '-')}"] + ([] if value is True else [str(value)]))
    return parser.parse_args(argv)


def test_command_is_registered():
    assert COMMANDS["attribution"] is cmd_attribution


def test_runs_and_reports_effects(holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark)) == 0

    output = capsys.readouterr().out
    assert "Brinson Attribution" in output
    assert "Allocation:" in output
    assert "Total Active Return:" in output
    assert "Tech" in output


def test_json_effects_sum_to_active_return(holdings):
    portfolio, benchmark = holdings

    args = _args(portfolio, benchmark, json=True)
    assert cmd_attribution(args) == 0


def test_json_payload_matches_hand_calculation(holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    active = (0.7 * 0.12 + 0.3 * 0.02) - (0.5 * 0.10 + 0.5 * 0.03)
    assert payload["total_active_return"] == pytest.approx(active)
    assert payload["total_active_return"] == pytest.approx(
        payload["allocation_effect"]
        + payload["selection_effect"]
        + payload["interaction_effect"]
    )
    assert {entry["sector"] for entry in payload["sectors"]} == {"Tech", "Energy"}


def test_sectors_are_ranked_worst_first(holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark, json=True)) == 0
    totals = [entry["total"] for entry in json.loads(capsys.readouterr().out)["sectors"]]

    assert totals == sorted(totals)


def test_top_limits_sector_rows(holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark, top=1, json=True)) == 0
    payload = json.loads(capsys.readouterr().out)

    assert len(payload["sectors"]) == 1


def test_rejects_non_positive_top(holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark, top=0)) == 1
    assert "--top must be at least 1" in capsys.readouterr().out


def test_missing_portfolio_file(tmp_path, holdings, capsys):
    _, benchmark = holdings

    assert cmd_attribution(_args(tmp_path / "nope.parquet", benchmark)) == 1
    assert "portfolio file not found" in capsys.readouterr().out


def test_missing_benchmark_file(tmp_path, holdings, capsys):
    portfolio, _ = holdings

    assert cmd_attribution(_args(portfolio, tmp_path / "nope.parquet")) == 1
    assert "benchmark file not found" in capsys.readouterr().out


def test_missing_column_is_reported(tmp_path, holdings, capsys):
    portfolio, benchmark = holdings

    assert cmd_attribution(_args(portfolio, benchmark, sector_col="industry")) == 1
    output = capsys.readouterr().out
    assert "Column 'industry' not found" in output
    assert "Available columns" in output


def test_all_null_rows_are_rejected(tmp_path, holdings, capsys):
    _, benchmark = holdings
    empty = tmp_path / "empty.parquet"
    pl.DataFrame({
        "sector": [None],
        "weight": [None],
        "return": [None],
    }, schema={"sector": pl.Utf8, "weight": pl.Float64, "return": pl.Float64}).write_parquet(empty)

    assert cmd_attribution(_args(empty, benchmark)) == 1
    assert "no complete rows" in capsys.readouterr().out


def test_custom_column_names(tmp_path, capsys):
    schema = {"gics": ["Tech"], "w": [1.0], "r": [0.10]}
    portfolio = tmp_path / "p.parquet"
    benchmark = tmp_path / "b.parquet"
    pl.DataFrame(schema).write_parquet(portfolio)
    pl.DataFrame({"gics": ["Tech"], "w": [1.0], "r": [0.08]}).write_parquet(benchmark)

    parser = create_parser()
    args = parser.parse_args([
        "attribution",
        "--portfolio", str(portfolio),
        "--benchmark", str(benchmark),
        "--sector-col", "gics",
        "--weight-col", "w",
        "--return-col", "r",
        "--json",
    ])

    assert cmd_attribution(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["selection_effect"] == pytest.approx(0.02)
    assert payload["total_active_return"] == pytest.approx(0.02)
