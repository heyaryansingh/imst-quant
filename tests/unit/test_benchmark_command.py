"""Tests for the `benchmark` CLI command."""

import json

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_benchmark, create_parser


def _run(capsys, strategy, *extra):
    args = create_parser().parse_args(["benchmark", "--strategy", str(strategy), "--json", *extra])
    code = cmd_benchmark(args)
    return code, capsys.readouterr().out


def test_aligns_on_dates_not_position(tmp_path, capsys):
    rng = np.random.default_rng(0)
    bench = rng.normal(0, 0.01, 100)
    pl.DataFrame({"date": np.arange(100), "returns": bench}).write_parquet(tmp_path / "b.parquet")
    # Strategy = 2x benchmark but missing the last 10 dates; a tail-based
    # alignment would pair it with the wrong benchmark bars.
    pl.DataFrame({"date": np.arange(90), "returns": 2 * bench[:90]}).write_parquet(tmp_path / "s.parquet")
    code, out = _run(capsys, tmp_path / "s.parquet", "--benchmark", str(tmp_path / "b.parquet"))
    assert code == 0
    payload = json.loads(out)
    assert payload["beta"] == pytest.approx(2.0)
    assert payload["correlation"] == pytest.approx(1.0)


def test_no_benchmark_keeps_json_clean(tmp_path, capsys):
    pl.DataFrame({"returns": [0.01, -0.02, 0.03, 0.0]}).write_parquet(tmp_path / "s.parquet")
    code, out = _run(capsys, tmp_path / "s.parquet")
    assert code == 0
    json.loads(out)


def test_benchmark_without_return_column(tmp_path, capsys):
    pl.DataFrame({"returns": [0.01, -0.02]}).write_parquet(tmp_path / "s.parquet")
    pl.DataFrame({"price": [1.0, 2.0]}).write_parquet(tmp_path / "b.parquet")
    code, out = _run(capsys, tmp_path / "s.parquet", "--benchmark", str(tmp_path / "b.parquet"))
    assert code == 1
    assert "Error" in out
