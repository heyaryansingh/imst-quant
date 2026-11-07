"""Tests for the `stopsweep` CLI command and simulate_trailing_stop."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_stopsweep, create_parser
from imst_quant.trading.trailing_stop import simulate_trailing_stop


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def test_stop_out_and_reenter():
    prices = [100, 110, 99, 90, 95, 100, 120]
    r = simulate_trailing_stop(prices, 0.05, cooldown=1)
    # Long 100->99, stopped at 99, flat through 90, re-enter at 90 -> 120.
    assert r["total_return"] == pytest.approx(0.99 * 120 / 90 - 1)
    assert r["n_stops"] == 1
    assert r["buy_hold_return"] == pytest.approx(0.2)
    assert r["time_in_market"] == pytest.approx(5 / 6)


def test_wide_stop_matches_buy_and_hold():
    prices = [100, 101, 99, 102, 104]
    r = simulate_trailing_stop(prices, 0.5)
    assert r["total_return"] == pytest.approx(r["buy_hold_return"])
    assert r["n_stops"] == 0


def test_rejects_bad_prices():
    with pytest.raises(ValueError):
        simulate_trailing_stop([100, 0, 101], 0.05)


def _run(tmp_path, capsys, prices, *extra):
    path = tmp_path / "px.parquet"
    pl.DataFrame({"close": prices}).write_parquet(path)
    args = create_parser().parse_args(["stopsweep", "--data", str(path), *extra])
    return cmd_stopsweep(args), capsys.readouterr().out


def test_cli_json(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [100, 110, 99, 90, 95, 100, 120], "--pcts", "0.05,0.5", "--cooldown", "1", "--json")
    payload = json.loads(out)
    assert code == 0
    assert [r["pct"] for r in payload["results"]] == [0.05, 0.5]
    assert payload["best_pct"] == 0.05


def test_cli_text_and_bad_pcts(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, [100, 101, 102])
    assert code == 0 and "Trailing Stop Sweep" in out
    code, out = _run(tmp_path, capsys, [100, 101, 102], "--pcts", "1.5")
    assert code == 1 and "between 0 and 1" in out
