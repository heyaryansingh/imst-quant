"""Tests for the `calendar` CLI command."""

import json
import sys
from datetime import date

import polars as pl
import pytest
import structlog

from imst_quant.cli import cmd_calendar, create_parser


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(tmp_path, capsys, frame, *extra):
    path = tmp_path / "returns.parquet"
    pl.DataFrame(frame).write_parquet(path)
    args = create_parser().parse_args(["calendar", "--data", str(path), *extra])
    code = cmd_calendar(args)
    return code, capsys.readouterr().out


FRAME = {
    "date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 2, 1), date(2025, 3, 4), date(2025, 3, 5)],
    "returns": [0.10, 0.10, -0.05, 0.02, float("nan")],
}


def test_calendar_compounds_by_month_and_year(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, FRAME, "--json")
    assert code == 0
    payload = json.loads(out)
    y24 = payload["years"]["2024"]
    assert y24["months"]["1"] == pytest.approx(0.21)
    assert y24["months"]["2"] == pytest.approx(-0.05)
    assert y24["year"] == pytest.approx(1.21 * 0.95 - 1)
    assert payload["years"]["2025"]["year"] == pytest.approx(0.02)
    assert payload["n_observations"] == 4
    assert payload["pct_positive_months"] == pytest.approx(2 / 3)
    assert payload["worst_month"] == pytest.approx(-0.05)


def test_calendar_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, FRAME)
    assert code == 0
    assert "Calendar Returns" in out
    assert "2025" in out


@pytest.mark.parametrize(
    "frame",
    [
        {"date": [date(2024, 1, 2)], "other": [0.1]},
        {"date": [date(2024, 1, 2)], "returns": [-1.5]},
        {"date": [None], "returns": [0.1]},
    ],
)
def test_calendar_rejects_bad_input(tmp_path, capsys, frame):
    code, out = _run(tmp_path, capsys, frame)
    assert code == 1
    assert "Error" in out
