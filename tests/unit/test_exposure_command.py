"""Tests for the `exposure` CLI command."""

import json

import polars as pl

from imst_quant.cli import cmd_exposure, create_parser


def _run(tmp_path, capsys, rows, *extra):
    path = tmp_path / "portfolio.parquet"
    pl.DataFrame(rows).write_parquet(path)
    args = create_parser().parse_args(["exposure", "--portfolio", str(path), *extra])
    code = cmd_exposure(args)
    return code, capsys.readouterr().out


ROWS = {
    "symbol": ["a", "b", "c", "a"],
    "weight": [0.3, 0.3, 0.2, 0.2],
    "sector": ["t", None, "f", "t"],
    "asset_type": ["e"] * 4,
}


def test_duplicate_symbols_are_one_position(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, ROWS, "--json")
    assert code == 0
    p = json.loads(out)
    assert p["max_single_exposure"] == 0.5
    assert abs(p["herfindahl_index"] - 0.38) < 1e-9


def test_null_sector_text_report(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, ROWS)
    assert code == 0
    assert "Unknown" in out
