"""Tests for the `quality-check` CLI command."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import structlog


from imst_quant.cli import cmd_quality_check, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, monkeypatch, frame, *extra):
    data = SimpleNamespace(bronze_dir=tmp_path, silver_dir=tmp_path, gold_dir=tmp_path)
    monkeypatch.setattr(
        "imst_quant.config.settings.Settings", lambda: SimpleNamespace(data=data)
    )
    frame.write_parquet(tmp_path / "gold.parquet")
    args = create_parser().parse_args(["quality-check", "--layer", "gold", *extra])
    code = cmd_quality_check(args)
    return code, capsys.readouterr().out


GOOD = pl.DataFrame({"return_1d": np.random.default_rng(2).normal(0, 0.01, 60)})
BAD = pl.DataFrame(
    {
        "open": [10.0, 10.0],
        "high": [9.0, 11.0],
        "low": [9.5, 9.0],
        "close": [10.0, 10.0],
    }
)


def test_quality_check_clean_file_json(tmp_path, capsys, monkeypatch):
    code, out = _run(tmp_path, capsys, monkeypatch, GOOD, "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["files_checked"] == 1
    assert payload["issues_count"] == 0


def test_quality_check_flags_bad_ohlcv(tmp_path, capsys, monkeypatch):
    code, out = _run(tmp_path, capsys, monkeypatch, BAD, "--json")
    assert code == 1
    payload = json.loads(out)
    assert payload["issues_count"] == 1
    assert any("High < Low" in m for m in payload["issues"][0]["issues"])


def test_quality_check_text(tmp_path, capsys, monkeypatch):
    code, out = _run(tmp_path, capsys, monkeypatch, GOOD)
    assert code == 0
    assert "All files passed" in out
