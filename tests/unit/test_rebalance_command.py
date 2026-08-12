"""Tests for the `rebalance` CLI command."""

import json

import polars as pl
import pytest

from imst_quant.cli import cmd_rebalance, create_parser


def _weights_file(tmp_path, symbols, weights, name):
    path = tmp_path / name
    pl.DataFrame({"symbol": symbols, "weight": weights}).write_parquet(path)
    return path


def _run(tmp_path, capsys, current, target, extra=()):
    current_path = _weights_file(tmp_path, *current, name="current.parquet")
    target_path = _weights_file(tmp_path, *target, name="target.parquet")
    args = create_parser().parse_args([
        "rebalance",
        "--portfolio", str(current_path),
        "--target", str(target_path),
        "--json",
        *extra,
    ])
    assert cmd_rebalance(args) == 0
    return json.loads(capsys.readouterr().out)


def test_rebalance_command_is_registered():
    args = create_parser().parse_args(
        ["rebalance", "--portfolio", "a.parquet", "--target", "b.parquet"]
    )

    assert args.command == "rebalance"
    assert args.symbol_col == "symbol"
    assert args.weight_col == "weight"
    assert args.threshold == 0.05
    assert args.method == "absolute"
    assert args.min_trade == 100.0
    assert args.cost_bps == 0.0
    assert args.value is None


def test_drift_and_actions_are_reported(tmp_path, capsys):
    payload = _run(
        tmp_path,
        capsys,
        (["AAPL", "MSFT"], [0.60, 0.40]),
        (["AAPL", "MSFT"], [0.50, 0.50]),
    )

    assert payload["n_positions"] == 2
    assert payload["needs_rebalancing"] is True
    assert payload["max_absolute_drift"] == pytest.approx(0.10)
    # 10% overweight AAPL funded by 10% underweight MSFT trades 10% of the book.
    assert payload["turnover"] == pytest.approx(0.10)

    by_symbol = {p["symbol"]: p for p in payload["positions"]}
    assert by_symbol["AAPL"]["action"] == "SELL"
    assert by_symbol["MSFT"]["action"] == "BUY"
    assert by_symbol["AAPL"]["absolute_drift"] == pytest.approx(0.10)


def test_drift_inside_threshold_holds(tmp_path, capsys):
    payload = _run(
        tmp_path,
        capsys,
        (["AAPL", "MSFT"], [0.51, 0.49]),
        (["AAPL", "MSFT"], [0.50, 0.50]),
    )

    assert payload["needs_rebalancing"] is False
    assert {p["action"] for p in payload["positions"]} == {"SELL", "BUY"}


def test_relative_method_flags_small_positions_absolute_would_miss(tmp_path, capsys):
    # 1% absolute drift on a 2% target is a 50% relative miss.
    current = (["AAPL", "CASH"], [0.03, 0.97])
    target = (["AAPL", "CASH"], [0.02, 0.98])

    absolute = _run(tmp_path, capsys, current, target)
    relative = _run(tmp_path, capsys, current, target, extra=["--method", "relative"])

    assert absolute["needs_rebalancing"] is False
    assert relative["needs_rebalancing"] is True


def test_value_sizes_trades_and_estimates_cost(tmp_path, capsys):
    payload = _run(
        tmp_path,
        capsys,
        (["AAPL", "MSFT"], [0.60, 0.40]),
        (["AAPL", "MSFT"], [0.50, 0.50]),
        extra=["--value", "100000", "--cost-bps", "10"],
    )

    by_symbol = {p["symbol"]: p for p in payload["positions"]}
    # Overweight positions are sold, so the trade value is negative.
    assert by_symbol["AAPL"]["trade_value"] == pytest.approx(-10_000)
    assert by_symbol["MSFT"]["trade_value"] == pytest.approx(10_000)

    assert payload["n_trades"] == 2
    assert payload["traded_notional"] == pytest.approx(20_000)
    assert payload["estimated_cost"] == pytest.approx(20_000 * 0.001)


def test_min_trade_filters_dust(tmp_path, capsys):
    payload = _run(
        tmp_path,
        capsys,
        (["AAPL", "MSFT"], [0.5001, 0.4999]),
        (["AAPL", "MSFT"], [0.50, 0.50]),
        extra=["--value", "100000", "--min-trade", "100"],
    )

    # Each side trades $10, well under the $100 minimum.
    assert payload["n_trades"] == 0
    assert payload["skipped_trades"] == 2
    assert payload["traded_notional"] == pytest.approx(0.0)


def test_symbol_only_in_target_is_a_full_buy(tmp_path, capsys):
    payload = _run(
        tmp_path,
        capsys,
        (["AAPL"], [1.0]),
        (["AAPL", "NVDA"], [0.8, 0.2]),
        extra=["--value", "100000"],
    )

    by_symbol = {p["symbol"]: p for p in payload["positions"]}
    assert by_symbol["NVDA"]["current_weight"] == pytest.approx(0.0)
    assert by_symbol["NVDA"]["action"] == "BUY"
    assert by_symbol["NVDA"]["trade_value"] == pytest.approx(20_000)


def test_missing_target_file_errors(tmp_path, capsys):
    current_path = _weights_file(tmp_path, ["AAPL"], [1.0], name="current.parquet")
    args = create_parser().parse_args([
        "rebalance", "--portfolio", str(current_path),
        "--target", str(tmp_path / "nope.parquet"),
    ])

    assert cmd_rebalance(args) == 1
    assert "not found" in capsys.readouterr().out


def test_missing_column_errors(tmp_path, capsys):
    current_path = tmp_path / "current.parquet"
    pl.DataFrame({"ticker": ["AAPL"], "weight": [1.0]}).write_parquet(current_path)
    target_path = _weights_file(tmp_path, ["AAPL"], [1.0], name="target.parquet")

    args = create_parser().parse_args([
        "rebalance", "--portfolio", str(current_path), "--target", str(target_path),
    ])

    assert cmd_rebalance(args) == 1
    assert "not found in current portfolio file" in capsys.readouterr().out


def test_non_positive_value_errors(tmp_path, capsys):
    current_path = _weights_file(tmp_path, ["AAPL"], [1.0], name="current.parquet")
    target_path = _weights_file(tmp_path, ["AAPL"], [1.0], name="target.parquet")

    args = create_parser().parse_args([
        "rebalance", "--portfolio", str(current_path),
        "--target", str(target_path), "--value", "0",
    ])

    assert cmd_rebalance(args) == 1
    assert "--value must be positive" in capsys.readouterr().out


def test_text_output_flags_weights_that_do_not_sum_to_one(tmp_path, capsys):
    current_path = _weights_file(tmp_path, ["AAPL"], [0.5], name="current.parquet")
    target_path = _weights_file(tmp_path, ["AAPL"], [0.5], name="target.parquet")

    args = create_parser().parse_args([
        "rebalance", "--portfolio", str(current_path), "--target", str(target_path),
    ])
    assert cmd_rebalance(args) == 0

    out = capsys.readouterr().out
    assert "current weights sum to 50.00%" in out
    assert "HOLD" in out
