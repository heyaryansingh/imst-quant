"""Tests for the `execution` CLI command."""

import json
import math

from imst_quant.cli import cmd_execution, create_parser

HEADER = "trade_id,symbol,side,decision_price,order_price,fill_price,quantity_ordered,quantity_filled,commission,venue\n"


def _run(tmp_path, capsys, rows):
    path = tmp_path / "trades.csv"
    path.write_text(HEADER + rows)
    args = create_parser().parse_args(["execution", "--trades", str(path), "--json"])
    code = cmd_execution(args)
    return code, json.loads(capsys.readouterr().out)


def test_slippage_sign_buy_vs_sell(tmp_path, capsys):
    rows = "1,A,buy,100,100,100.10,100,100,0,X\n2,A,sell,100,100,100.10,100,100,0,Y\n"
    code, out = _run(tmp_path, capsys, rows)
    assert code == 0
    assert math.isclose(out["slippage_by_venue"]["X"], 10.0, abs_tol=1e-6)  # paid up
    assert math.isclose(out["slippage_by_venue"]["Y"], -10.0, abs_tol=1e-6)  # sold higher


def test_blank_commission_and_venue_not_nan(tmp_path, capsys):
    rows = "1,A,buy,100,100,100.10,100,50,,\n2,A,sell,100,100,99.90,100,100,1.0,X\n"
    code, out = _run(tmp_path, capsys, rows)
    assert code == 0
    assert out["commission_cost"] == 1.0
    assert "nan" not in out["slippage_by_venue"]
    assert "unknown" in out["slippage_by_venue"]


def test_empty_size_buckets_omitted(tmp_path, capsys):
    rows = "1,A,buy,100,100,100.10,100,100,0,X\n2,A,buy,100,100,100.10,100,100,0,X\n"
    _, out = _run(tmp_path, capsys, rows)
    assert set(out["slippage_by_size_bucket"]) == {"small"}
