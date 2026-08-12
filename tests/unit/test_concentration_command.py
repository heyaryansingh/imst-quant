"""Tests for the `concentration` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_concentration, create_parser

EQUAL_BOOK = [("AAPL", 0.25), ("MSFT", 0.25), ("NVDA", 0.25), ("TSLA", 0.25)]
LOPSIDED_BOOK = [("AAPL", 0.85), ("MSFT", 0.05), ("NVDA", 0.05), ("TSLA", 0.05)]


def _portfolio_file(tmp_path, rows, name="portfolio.parquet"):
    path = tmp_path / name
    pl.DataFrame(
        {"symbol": [r[0] for r in rows], "weight": [r[1] for r in rows]}
    ).write_parquet(path)
    return path


def test_concentration_command_is_registered():
    args = create_parser().parse_args(["concentration", "--portfolio", "x.parquet"])

    assert args.command == "concentration"
    assert args.top_n == 5
    assert args.weight_col == "weight"
    assert args.max_hhi is None


def test_concentration_command_reports_json(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--top-n", "2", "--json"]
    )
    assert cmd_concentration(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_positions"] == 4
    assert payload["hhi"] == pytest.approx(0.25)
    assert payload["effective_n"] == pytest.approx(4.0)
    assert payload["top_2_concentration"] == pytest.approx(0.5)
    assert payload["gross_exposure"] == pytest.approx(1.0)
    assert payload["has_shorts"] is False


def test_concentration_command_lists_the_largest_positions(tmp_path, capsys):
    path = _portfolio_file(tmp_path, LOPSIDED_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--top-n", "1", "--json"]
    )
    assert cmd_concentration(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [p["symbol"] for p in payload["top_positions"]] == ["AAPL"]
    assert payload["top_positions"][0]["share"] == pytest.approx(0.85)
    assert payload["hhi"] > 0.7


def test_concentration_command_uses_gross_weights_for_shorts(tmp_path, capsys):
    """A short leg must not push HHI above 1 by shrinking the signed total."""
    path = _portfolio_file(tmp_path, [("AAPL", 0.75), ("MSFT", 0.75), ("SPY", -0.5)])

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--json"]
    )
    assert cmd_concentration(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["has_shorts"] is True
    assert payload["n_positions"] == 3
    assert payload["gross_exposure"] == pytest.approx(2.0)
    assert 0 < payload["hhi"] <= 1.0
    assert payload["effective_n"] > 0
    assert payload["top_positions"][-1]["symbol"] == "SPY"


def test_concentration_command_drops_zero_weight_rows(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK + [("CLOSED", 0.0)])

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--json"]
    )
    assert cmd_concentration(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["n_positions"] == 4
    assert "CLOSED" not in [p["symbol"] for p in payload["top_positions"]]


def test_concentration_command_gates_on_max_hhi(tmp_path, capsys):
    path = _portfolio_file(tmp_path, LOPSIDED_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--max-hhi", "0.3"]
    )
    assert cmd_concentration(args) == 2
    assert "exceeds the --max-hhi limit" in capsys.readouterr().out


def test_concentration_command_passes_an_unbreached_gate(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--max-hhi", "0.3", "--json"]
    )
    assert cmd_concentration(args) == 0
    assert json.loads(capsys.readouterr().out)["breached"] is False


def test_concentration_command_prints_a_table(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(["concentration", "--portfolio", str(path)])
    assert cmd_concentration(args) == 0

    out = capsys.readouterr().out
    assert "Portfolio Concentration" in out
    assert "Effective N" in out
    assert "AAPL" in out


def test_concentration_command_rejects_a_bad_top_n(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--top-n", "0"]
    )

    assert cmd_concentration(args) == 1
    assert "--top-n" in capsys.readouterr().out


def test_concentration_command_rejects_a_bad_max_hhi(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--max-hhi", "1.5"]
    )

    assert cmd_concentration(args) == 1
    assert "--max-hhi" in capsys.readouterr().out


def test_concentration_command_reports_missing_column(tmp_path, capsys):
    path = _portfolio_file(tmp_path, EQUAL_BOOK)

    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(path), "--weight-col", "target"]
    )

    assert cmd_concentration(args) == 1
    assert "not found" in capsys.readouterr().out


def test_concentration_command_reports_an_empty_book(tmp_path, capsys):
    path = _portfolio_file(tmp_path, [("AAPL", 0.0)])

    args = create_parser().parse_args(["concentration", "--portfolio", str(path)])

    assert cmd_concentration(args) == 1
    assert "non-zero weight" in capsys.readouterr().out


def test_concentration_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(
        ["concentration", "--portfolio", str(tmp_path / "nope.parquet")]
    )

    assert cmd_concentration(args) == 1
    assert "not found" in capsys.readouterr().out
