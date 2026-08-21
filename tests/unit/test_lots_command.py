"""Tests for the `lots` CLI command."""

import datetime as dt
import json

import polars as pl

from imst_quant.cli import cmd_lots, create_parser

# Two lots bought at different prices, then a sale that spans both.
TRADES = [
    {"date": dt.date(2023, 1, 10), "symbol": "AAPL", "side": "buy",
     "quantity": 100.0, "price": 150.0, "commission": 0.0},
    {"date": dt.date(2024, 6, 10), "symbol": "AAPL", "side": "buy",
     "quantity": 100.0, "price": 170.0, "commission": 0.0},
    {"date": dt.date(2024, 8, 10), "symbol": "AAPL", "side": "sell",
     "quantity": 150.0, "price": 180.0, "commission": 0.0},
]


def _trades_file(tmp_path, trades=TRADES, name="trade_history.parquet"):
    path = tmp_path / name
    pl.DataFrame(trades).write_parquet(path)
    return path


def _prices_file(tmp_path, prices, name="prices.parquet"):
    path = tmp_path / name
    pl.DataFrame(
        [{"symbol": symbol, "price": price} for symbol, price in prices.items()]
    ).write_parquet(path)
    return path


def _run(argv):
    return cmd_lots(create_parser().parse_args(argv))


def test_lots_command_is_registered():
    args = create_parser().parse_args(["lots", "--trades", "x.parquet"])

    assert args.command == "lots"
    assert args.method == "fifo"
    assert args.date_col == "date"
    assert args.min_loss == 0.0
    assert not args.compare


def test_fifo_report_splits_short_and_long_term(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "fifo"
    assert payload["sale_count"] == 2
    assert payload["realized_pnl"] == 3500.0
    # 100 shares held since 2023 are long term; the 50 from June are not.
    assert payload["long_term_pnl"] == 3000.0
    assert payload["short_term_pnl"] == 500.0
    assert payload["open_lots"] == 1


def test_lifo_matches_the_newest_lot_first(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--method", "lifo", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["realized_pnl"] == 2500.0
    assert payload["long_term_pnl"] == 1500.0


def test_compare_reports_every_method(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--compare", "--json"]) == 0

    comparison = json.loads(capsys.readouterr().out)["method_comparison"]
    assert set(comparison) == {"fifo", "lifo", "hifo", "lofo"}
    assert comparison["hifo"]["realized_pnl"] <= comparison["lofo"]["realized_pnl"]


def test_text_output_renders(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path)]) == 0

    out = capsys.readouterr().out
    assert "Tax Lots (FIFO)" in out
    assert "Realized P&L" in out


def test_prices_add_unrealized_pnl(tmp_path, capsys):
    trades = _trades_file(tmp_path)
    prices = _prices_file(tmp_path, {"AAPL": 200.0})

    assert _run(
        ["lots", "--trades", str(trades), "--prices", str(prices), "--json"]
    ) == 0

    payload = json.loads(capsys.readouterr().out)
    # 50 shares left at a 170 basis, marked at 200.
    assert payload["unrealized_pnl"] == 1500.0


def test_harvest_lists_losing_lots(tmp_path, capsys):
    trades = _trades_file(tmp_path)
    prices = _prices_file(tmp_path, {"AAPL": 100.0})

    assert _run(
        ["lots", "--trades", str(trades), "--prices", str(prices),
         "--harvest", "--json"]
    ) == 0

    lots = json.loads(capsys.readouterr().out)["harvestable_lots"]
    assert len(lots) == 1
    assert lots[0]["symbol"] == "AAPL"
    assert lots[0]["unrealized_loss"] == -3500.0


def test_harvest_without_prices_is_rejected(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--harvest"]) == 1
    assert "--harvest needs --prices" in capsys.readouterr().out


def test_negative_min_loss_is_rejected(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--min-loss", "-1"]) == 1
    assert "--min-loss must be non-negative" in capsys.readouterr().out


def test_missing_trade_file_is_reported(tmp_path, capsys):
    assert _run(["lots", "--trades", str(tmp_path / "nope.parquet")]) == 1
    assert "Trade history file not found" in capsys.readouterr().out


def test_missing_column_is_reported(tmp_path, capsys):
    path = tmp_path / "bad.parquet"
    pl.DataFrame([{"date": dt.date(2024, 1, 1), "symbol": "AAPL"}]).write_parquet(path)

    assert _run(["lots", "--trades", str(path)]) == 1
    assert "not found in trade history file" in capsys.readouterr().out


def test_unknown_trade_side_is_reported(tmp_path, capsys):
    trades = [dict(TRADES[0], side="short")]
    path = _trades_file(tmp_path, trades, name="odd.parquet")

    assert _run(["lots", "--trades", str(path)]) == 1
    assert "Unknown trade sides" in capsys.readouterr().out


def test_selling_more_than_held_is_reported(tmp_path, capsys):
    trades = [TRADES[0], dict(TRADES[2], quantity=500.0)]
    path = _trades_file(tmp_path, trades, name="oversold.parquet")

    assert _run(["lots", "--trades", str(path)]) == 1
    assert "only" in capsys.readouterr().out


def test_symbol_filter_restricts_the_report(tmp_path, capsys):
    trades = TRADES + [
        {"date": dt.date(2024, 1, 5), "symbol": "MSFT", "side": "buy",
         "quantity": 10.0, "price": 100.0, "commission": 0.0},
        {"date": dt.date(2024, 2, 5), "symbol": "MSFT", "side": "sell",
         "quantity": 10.0, "price": 300.0, "commission": 0.0},
    ]
    path = _trades_file(tmp_path, trades, name="multi.parquet")

    assert _run(["lots", "--trades", str(path), "--symbol", "AAPL", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["realized_pnl"] == 3500.0


def test_unknown_symbol_filter_is_reported(tmp_path, capsys):
    path = _trades_file(tmp_path)

    assert _run(["lots", "--trades", str(path), "--symbol", "TSLA"]) == 1
    assert "No trades found for symbol" in capsys.readouterr().out


def test_commission_column_is_optional(tmp_path, capsys):
    trades = [
        {key: value for key, value in trade.items() if key != "commission"}
        for trade in TRADES
    ]
    path = _trades_file(tmp_path, trades, name="no_commission.parquet")

    assert _run(["lots", "--trades", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["realized_pnl"] == 3500.0


def test_trades_are_replayed_in_date_order(tmp_path, capsys):
    shuffled = [TRADES[2], TRADES[0], TRADES[1]]
    path = _trades_file(tmp_path, shuffled, name="shuffled.parquet")

    assert _run(["lots", "--trades", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["realized_pnl"] == 3500.0
