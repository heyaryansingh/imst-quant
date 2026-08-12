"""Tests for the `indicators` CLI command."""

import datetime as dt
import json
import math

import polars as pl

from imst_quant.cli import cmd_indicators, create_parser

N_BARS = 80


def _closes(n=N_BARS):
    """A gently trending series with enough wiggle to move an oscillator."""
    return [100 + i * 0.5 + 3 * math.sin(i / 3) for i in range(n)]


def _prices_file(tmp_path, *, ohlc=True, volume=True, symbols=None, n=N_BARS):
    closes = _closes(n)
    frame = {
        "date": [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(n)],
        "close": closes,
    }
    if ohlc:
        frame["high"] = [c + 1.0 for c in closes]
        frame["low"] = [c - 1.0 for c in closes]
    if volume:
        frame["volume"] = [1_000_000 + i * 1_000 for i in range(n)]
    df = pl.DataFrame(frame)

    if symbols:
        df = pl.concat(
            [df.with_columns(pl.lit(symbol).alias("symbol")) for symbol in symbols]
        )

    path = tmp_path / "prices.parquet"
    df.write_parquet(path)
    return path


def test_indicators_command_is_registered():
    args = create_parser().parse_args(["indicators", "--prices", "x.parquet"])

    assert args.command == "indicators"
    assert args.rsi_period == 14
    assert args.bb_window == 20
    assert args.close_col == "close"
    assert args.symbol is None


def test_indicators_command_reports_json(tmp_path, capsys):
    path = _prices_file(tmp_path)

    args = create_parser().parse_args(["indicators", "--prices", str(path), "--json"])
    assert cmd_indicators(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["bars"] == N_BARS
    assert payload["skipped"] == []
    assert set(payload["indicators"]) == {
        "rsi", "macd", "bollinger_bands", "atr", "adx",
        "stochastic_oscillator", "obv", "vwap",
    }
    assert 0.0 <= payload["latest"]["rsi"] <= 100.0
    assert "macd_line" in payload["latest"]
    assert "date" not in payload["latest"]


def test_indicators_command_skips_what_the_columns_cannot_support(tmp_path, capsys):
    path = _prices_file(tmp_path, ohlc=False, volume=False)

    args = create_parser().parse_args(["indicators", "--prices", str(path), "--json"])
    assert cmd_indicators(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["indicators"] == ["rsi", "macd", "bollinger_bands"]
    assert len(payload["skipped"]) == 2
    assert "atr" not in payload["latest"]


def test_indicators_command_computes_obv_without_ohlc(tmp_path, capsys):
    path = _prices_file(tmp_path, ohlc=False, volume=True)

    args = create_parser().parse_args(["indicators", "--prices", str(path), "--json"])
    assert cmd_indicators(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert "obv" in payload["indicators"]
    assert "vwap" not in payload["indicators"]


def test_indicators_command_writes_a_parquet(tmp_path, capsys):
    path = _prices_file(tmp_path)
    out = tmp_path / "nested" / "indicators.parquet"

    args = create_parser().parse_args(
        ["indicators", "--prices", str(path), "--output", str(out), "--json"]
    )
    assert cmd_indicators(args) == 0

    written = pl.read_parquet(out)
    assert written.height == N_BARS
    assert "rsi" in written.columns
    assert json.loads(capsys.readouterr().out)["output"] == str(out)


def test_indicators_command_filters_to_one_symbol(tmp_path, capsys):
    path = _prices_file(tmp_path, symbols=["AAPL", "MSFT"])

    args = create_parser().parse_args(
        ["indicators", "--prices", str(path), "--symbol", "MSFT", "--json"]
    )
    assert cmd_indicators(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["bars"] == N_BARS
    assert payload["symbol"] == "MSFT"


def test_indicators_command_refuses_a_multi_symbol_file(tmp_path, capsys):
    """Rolling windows would span the seam between two symbols' bars."""
    path = _prices_file(tmp_path, symbols=["AAPL", "MSFT"])

    args = create_parser().parse_args(["indicators", "--prices", str(path)])

    assert cmd_indicators(args) == 1
    assert "--symbol" in capsys.readouterr().out


def test_indicators_command_reports_an_unknown_symbol(tmp_path, capsys):
    path = _prices_file(tmp_path, symbols=["AAPL"])

    args = create_parser().parse_args(
        ["indicators", "--prices", str(path), "--symbol", "NVDA"]
    )

    assert cmd_indicators(args) == 1
    assert "No rows for symbol" in capsys.readouterr().out


def test_indicators_command_prints_a_table(tmp_path, capsys):
    path = _prices_file(tmp_path)

    args = create_parser().parse_args(["indicators", "--prices", str(path)])
    assert cmd_indicators(args) == 0

    out = capsys.readouterr().out
    assert "Technical Indicators" in out
    assert "rsi" in out
    assert "Last Close" in out


def test_indicators_command_rejects_too_short_a_series(tmp_path, capsys):
    path = _prices_file(tmp_path, n=20)

    args = create_parser().parse_args(["indicators", "--prices", str(path)])

    assert cmd_indicators(args) == 1
    assert "Need more than" in capsys.readouterr().out


def test_indicators_command_rejects_a_degenerate_period(tmp_path, capsys):
    path = _prices_file(tmp_path)

    args = create_parser().parse_args(
        ["indicators", "--prices", str(path), "--rsi-period", "1"]
    )

    assert cmd_indicators(args) == 1
    assert "--rsi-period" in capsys.readouterr().out


def test_indicators_command_reports_missing_column(tmp_path, capsys):
    path = _prices_file(tmp_path)

    args = create_parser().parse_args(
        ["indicators", "--prices", str(path), "--close-col", "px"]
    )

    assert cmd_indicators(args) == 1
    assert "not found" in capsys.readouterr().out


def test_indicators_command_reports_missing_file(tmp_path, capsys):
    args = create_parser().parse_args(
        ["indicators", "--prices", str(tmp_path / "nope.parquet")]
    )

    assert cmd_indicators(args) == 1
    assert "not found" in capsys.readouterr().out
