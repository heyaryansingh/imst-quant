"""Tests for the `comoments` CLI command."""

import json

import numpy as np
import polars as pl

from imst_quant.cli import cmd_comoments, create_parser


def _file(tmp_path, strat, bench):
    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": strat, "benchmark_returns": bench}).write_parquet(path)
    return path


def _run(path, capsys, *extra):
    args = create_parser().parse_args(
        ["comoments", "--returns", str(path), "--json", *extra]
    )
    code = cmd_comoments(args)
    out = capsys.readouterr().out
    return code, (json.loads(out) if code == 0 else out)


def test_comoments_registered():
    args = create_parser().parse_args(["comoments"])
    assert args.command == "comoments"
    assert args.threshold == 0.0


def test_comoments_detects_crash_exposure(tmp_path, capsys):
    rng = np.random.default_rng(0)
    bench = rng.normal(0, 0.01, 300)
    # Beta 2 when the benchmark falls, 0.5 when it rises.
    strat = np.where(bench < 0, 2 * bench, 0.5 * bench) + rng.normal(0, 1e-4, 300)
    code, out = _run(_file(tmp_path, strat, bench), capsys)
    assert code == 0
    assert abs(out["downside_beta"] - 2) < 0.1
    assert abs(out["upside_beta"] - 0.5) < 0.1
    assert out["assessment"] == "crash_exposed"
    assert out["n_observations"] == 300


def test_comoments_rejects_bad_input(tmp_path, capsys):
    code, out = _run(tmp_path / "missing.parquet", capsys)
    assert code == 1 and "not found" in out

    code, out = _run(_file(tmp_path, [0.01] * 5, [0.02] * 5), capsys)
    assert code == 1 and "at least" in out

    code, out = _run(_file(tmp_path, [0.01] * 30, [0.0] * 30), capsys)
    assert code == 1
