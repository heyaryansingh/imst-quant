"""Tests for the `seasonality` CLI command."""

from datetime import datetime, timedelta

import json
import sys

import numpy as np
import polars as pl
import pytest
import structlog


from imst_quant.cli import cmd_seasonality, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, *extra):
    n = 300
    path = tmp_path / "features.parquet"
    pl.DataFrame(
        {
            "date": [datetime(2022, 1, 3) + timedelta(days=i) for i in range(n)],
            "return_1d": np.random.default_rng(5).normal(0.0, 0.01, n),
        }
    ).write_parquet(path)
    args = create_parser().parse_args(["seasonality", "--features", str(path), *extra])
    code = cmd_seasonality(args)
    return code, capsys.readouterr().out


@pytest.mark.parametrize("kind", ["all", "dow", "monthly", "tom", "holiday"])
def test_seasonality_json_every_type(tmp_path, capsys, kind):
    code, out = _run(tmp_path, capsys, "--type", kind, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload  # parseable, non-empty


def test_seasonality_dow_json_fields(tmp_path, capsys):
    _, out = _run(tmp_path, capsys, "--type", "dow", "--json")
    payload = json.loads(out)
    assert {"best", "worst", "spread", "significant"} <= set(payload)


def test_seasonality_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--type", "dow")
    assert code == 0
    assert "Day of Week Effect" in out
