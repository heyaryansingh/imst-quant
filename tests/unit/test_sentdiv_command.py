"""Tests for the `sentdiv` CLI command."""

import json
import sys

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_sentdiv, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, df, *extra):
    path = tmp_path / "s.parquet"
    df.write_parquet(path)
    code = cmd_sentdiv(create_parser().parse_args(["sentdiv", "--data", str(path), *extra]))
    return code, capsys.readouterr().out


def test_bullish_divergence_scored_against_forward_returns(tmp_path, capsys):
    # Sentiment falls while prices rise for 6 bars, then prices keep rising 1%/bar.
    sent = [0.5, 0.5, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1] + [-0.1] * 6
    rets = [0.0, 0.0, 0.0] + [0.01] * 12
    code, out = _run(tmp_path, capsys, pl.DataFrame({"sentiment": sent, "returns": rets}),
                     "--window", "2", "--horizon", "3", "--json")
    res = json.loads(out)
    assert code == 0
    assert res["bullish"]["count"] == 1
    assert res["bullish"]["avg_forward_return"] == pytest.approx(1.01 ** 3 - 1)
    assert res["bullish"]["hit_rate"] == 1.0
    assert res["bearish"]["count"] == 0 and res["bearish"]["hit_rate"] is None
    assert res["divergence_now"] == "none"  # sentiment flattened out: not a divergence


def test_text_output_and_bad_input(tmp_path, capsys):
    df = pl.DataFrame({"sentiment": [0.1 * i for i in range(20)], "returns": [-0.01] * 20})
    code, out = _run(tmp_path, capsys, df, "--window", "3")
    assert code == 0 and "BEARISH" in out
    assert _run(tmp_path, capsys, df, "--horizon", "0")[0] == 1
    assert _run(tmp_path, capsys, df, "--return-col", "nope")[0] == 1
    assert _run(tmp_path, capsys, df.head(3))[0] == 1
