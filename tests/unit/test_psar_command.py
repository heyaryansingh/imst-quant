"""Tests for parabolic_sar and the `psar` CLI command."""

import json
import math
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_psar, create_parser
from imst_quant.utils.technical_indicators import parabolic_sar


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


OHLC = {
    "high": [10.0, 11, 12, 13, 12, 10, 9],
    "low": [9.0, 10, 11, 12, 10, 8, 7],
    "close": [9.5, 10.5, 11.5, 12.5, 11, 9, 8],
}


def test_sar_accelerates_then_reverses_by_hand():
    out = parabolic_sar(pl.DataFrame(OHLC))
    assert out["psar_direction"].to_list() == [None, 1, 1, 1, 1, -1, -1]
    sar = out["psar"].to_list()
    assert sar[0] is None
    assert sar[1:] == pytest.approx([9.0, 9.0, 9.18, 9.4856, 13.0, 12.9])
    assert out["psar_af"].to_list()[1:] == pytest.approx([0.04, 0.06, 0.08, 0.08, 0.02, 0.04])


def test_af_capped_and_bad_params():
    df = pl.DataFrame({"high": [float(i) for i in range(1, 30)], "low": [i - 0.5 for i in range(1, 30)]})
    assert max(parabolic_sar(df, step=0.05, max_step=0.1)["psar_af"].drop_nulls()) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        parabolic_sar(df, step=0)
    with pytest.raises(ValueError):
        parabolic_sar(df, step=0.1, max_step=0.05)


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "ohlc.parquet"
    df.write_parquet(path)
    args = create_parser().parse_args(["psar", "--data", str(path), *extra])
    code = cmd_psar(args)
    return code, capsys.readouterr().out


def test_cli_json_reports_downtrend_and_skips_bad_bars(tmp_path, capsys):
    rows = {k: v[:2] + [math.nan if k == "low" else 5.0] + v[2:] for k, v in OHLC.items()}
    code, out = _run(tmp_path, capsys, pl.DataFrame(rows), "--json")
    assert code == 0
    res = json.loads(out)
    assert res["n_bars"] == 7
    assert res["trend"] == "down"
    assert res["psar"] == pytest.approx(12.9)
    assert res["bars_in_trend"] == 2
    assert res["flips"] == 1


def test_cli_errors(tmp_path, capsys):
    df = pl.DataFrame(OHLC)
    assert _run(tmp_path, capsys, df.head(1))[0] == 1
    assert _run(tmp_path, capsys, df.drop("close"))[0] == 1
    assert _run(tmp_path, capsys, df, "--step", "0.3")[0] == 1
