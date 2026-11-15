"""Tests for pivot_points and the `pivots` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_pivots, create_parser
from imst_quant.utils.technical_indicators import pivot_points


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


BARS = pl.DataFrame({"high": [12.0, 20], "low": [9.0, 10], "close": [9.0, 15]})


def test_classic_levels_use_previous_bar():
    row = pivot_points(BARS).row(1, named=True)
    # P = 10, R = 3
    assert [row[k] for k in ("s3", "s2", "s1", "pivot", "r1", "r2", "r3")] == pytest.approx(
        [9 - 2 * 2, 7, 8, 10, 11, 13, 12 + 2 * 1]
    )
    assert pivot_points(BARS).row(0, named=True)["pivot"] is None


def test_fibonacci_levels_and_bad_method():
    row = pivot_points(BARS, method="fibonacci").row(1, named=True)
    assert row["r1"] == pytest.approx(10 + 0.382 * 3)
    assert row["s2"] == pytest.approx(10 - 0.618 * 3)
    assert row["r3"] == pytest.approx(13)
    with pytest.raises(ValueError):
        pivot_points(BARS, method="camarilla")


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    code = cmd_pivots(create_parser().parse_args(["pivots", "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def test_cli_levels_come_from_last_bar(tmp_path, capsys):
    # Last bar H=20 L=10 C=15 -> P=15, R1=20, S1=10; close 15 sits on the pivot.
    code, out = _run(tmp_path, capsys, BARS, "--json")
    assert code == 0
    res = json.loads(out)
    assert res["levels"]["pivot"] == pytest.approx(15)
    assert res["levels"]["r1"] == pytest.approx(20) and res["levels"]["s1"] == pytest.approx(10)
    assert res["bias"] == "neutral"
    assert res["support"] == "s1" and res["resistance"] == "r1"
    assert res["support_distance_pct"] == pytest.approx(-1 / 3)


def test_cli_text_and_zero_range(tmp_path, capsys):
    df = pl.DataFrame({"high": [10.0, 18], "low": [9.0, 10], "close": [9.5, 17.5]})
    code, out = _run(tmp_path, capsys, df)
    assert code == 0 and "Bias: BULLISH" in out and "between pivot and r1" in out
    flat = pl.DataFrame({"high": [5.0], "low": [5.0], "close": [5.0]})
    code, out = _run(tmp_path, capsys, flat, "--json")
    assert code == 0 and json.loads(out)["zone"] == "at pivot"
