"""Tests for the `concentration` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_concentration, create_parser


def _portfolio_file(tmp_path, symbols, weights, name="portfolio.parquet"):
    path = tmp_path / name
    pl.DataFrame({"symbol": symbols, "weight": weights}).write_parquet(path)
    return path


def _run(tmp_path, capsys, symbols, weights, extra=(), expected_code=0):
    path = _portfolio_file(tmp_path, symbols, weights)
    args = create_parser().parse_args([
        "concentration", "--portfolio", str(path), "--json", *extra,
    ])
    assert cmd_concentration(args) == expected_code
    return json.loads(capsys.readouterr().out)


def test_concentration_command_is_registered():
    args = create_parser().parse_args(["concentration", "--portfolio", "x.parquet"])

    assert args.command == "concentration"
    assert args.top_n == 5
    assert args.max_weight is None
    assert args.min_effective_n is None


def test_equal_weights_are_maximally_diversified(tmp_path, capsys):
    payload = _run(tmp_path, capsys, ["A", "B", "C", "D"], [0.25] * 4)

    assert payload["hhi"] == pytest.approx(0.25)
    assert payload["effective_n"] == pytest.approx(4.0)
    assert payload["diversification_ratio"] == pytest.approx(1.0)
    assert payload["gini"] == pytest.approx(0.0, abs=1e-9)
    assert payload["normalized_entropy"] == pytest.approx(1.0)


def test_single_position_is_maximally_concentrated(tmp_path, capsys):
    payload = _run(tmp_path, capsys, ["AAPL"], [1.0])

    assert payload["hhi"] == pytest.approx(1.0)
    assert payload["effective_n"] == pytest.approx(1.0)
    assert payload["largest_symbol"] == "AAPL"


def test_top_n_concentration_uses_the_largest_positions(tmp_path, capsys):
    payload = _run(
        tmp_path, capsys, ["A", "B", "C", "D"], [0.4, 0.3, 0.2, 0.1],
        extra=["--top-n", "2"],
    )

    assert payload["top_2_concentration"] == pytest.approx(0.7)
    assert payload["largest_symbol"] == "A"
    assert payload["largest_weight"] == pytest.approx(0.4)


def test_short_positions_count_toward_concentration(tmp_path, capsys):
    # A 50/-50 book is a single-name bet twice over, not a diversified one.
    payload = _run(tmp_path, capsys, ["A", "B"], [0.5, -0.5])

    assert payload["gross_exposure"] == pytest.approx(1.0)
    assert payload["weight_sum"] == pytest.approx(0.0)
    assert payload["effective_n"] == pytest.approx(2.0)


def test_partly_invested_book_is_normalized_to_gross(tmp_path, capsys):
    payload = _run(tmp_path, capsys, ["A", "B"], [0.3, 0.3])

    assert payload["gross_exposure"] == pytest.approx(0.6)
    # Two half-sized but equal positions are still an effective N of 2.
    assert payload["effective_n"] == pytest.approx(2.0)


def test_max_weight_flags_oversized_positions(tmp_path, capsys):
    payload = _run(
        tmp_path, capsys, ["A", "B", "C"], [0.5, 0.3, 0.2],
        extra=["--max-weight", "0.35"],
    )

    assert [b["symbol"] for b in payload["max_weight_breaches"]] == ["A"]


def test_min_effective_n_breach_exits_two(tmp_path, capsys):
    payload = _run(
        tmp_path, capsys, ["A", "B"], [0.9, 0.1],
        extra=["--min-effective-n", "3"], expected_code=2,
    )

    assert payload["min_effective_n_breached"] is True


def test_min_effective_n_satisfied_exits_zero(tmp_path, capsys):
    payload = _run(
        tmp_path, capsys, ["A", "B", "C", "D"], [0.25] * 4,
        extra=["--min-effective-n", "3"],
    )

    assert payload["min_effective_n_breached"] is False


def test_all_zero_weights_errors(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["A", "B"], [0.0, 0.0])
    args = create_parser().parse_args(["concentration", "--portfolio", str(path)])

    assert cmd_concentration(args) == 1
    assert "all zero" in capsys.readouterr().out


def test_missing_column_errors(tmp_path, capsys):
    path = tmp_path / "portfolio.parquet"
    pl.DataFrame({"ticker": ["A"], "weight": [1.0]}).write_parquet(path)
    args = create_parser().parse_args(["concentration", "--portfolio", str(path)])

    assert cmd_concentration(args) == 1
    assert "not found in portfolio file" in capsys.readouterr().out


def test_invalid_max_weight_errors(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["A"], [1.0])
    args = create_parser().parse_args([
        "concentration", "--portfolio", str(path), "--max-weight", "1.5",
    ])

    assert cmd_concentration(args) == 1
    assert "--max-weight must be in (0, 1]" in capsys.readouterr().out


def test_text_output_lists_cumulative_weight(tmp_path, capsys):
    path = _portfolio_file(tmp_path, ["A", "B"], [0.6, 0.4])
    args = create_parser().parse_args([
        "concentration", "--portfolio", str(path), "--top-n", "2",
    ])
    assert cmd_concentration(args) == 0

    out = capsys.readouterr().out
    assert "Effective N" in out
    assert "100.00%" in out  # cumulative weight of both positions
