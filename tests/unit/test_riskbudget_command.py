"""Tests for the `riskbudget` CLI command."""

import json
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from imst_quant.cli import cmd_riskbudget, create_parser


def _run(tmp_path, capsys, *extra):
    rng = np.random.default_rng(0)
    n = 120
    dates = [date(2026, 1, 1) + timedelta(days=i) for i in range(n)]
    vols = {"LOW": 0.005, "MID": 0.01, "HIGH": 0.03}
    rows = [
        {"date": d, "asset_id": a, "return_1d": float(rng.normal(0, v))}
        for a, v in vols.items()
        for d in dates
    ]
    rows[5]["return_1d"] = float("nan")  # dropped, not propagated
    path = tmp_path / "features.parquet"
    pl.DataFrame(rows).write_parquet(path)
    args = create_parser().parse_args(["riskbudget", "--features", str(path), *extra])
    code = cmd_riskbudget(args)
    return code, capsys.readouterr().out


def test_riskbudget_equal_weight_shares_sum_to_one(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--json")
    assert code == 0
    payload = json.loads(out)
    assets = payload["assets"]
    assert sum(a["vol_share"] for a in assets.values()) == pytest.approx(1.0)
    assert sum(a["cvar_share"] for a in assets.values()) == pytest.approx(1.0)
    assert payload["top_risk_contributor"] == "HIGH"
    assert payload["observations"] == 119
    assert payload["diversification_ratio"] > 1.0


def test_riskbudget_custom_weights_are_normalised(tmp_path, capsys):
    code, out = _run(tmp_path, capsys, "--weights", "LOW=3,MID=1", "--json")
    assert code == 0
    assets = json.loads(out)["assets"]
    assert set(assets) == {"LOW", "MID"}
    assert assets["LOW"]["weight"] == pytest.approx(0.75)


def test_riskbudget_text(tmp_path, capsys):
    code, out = _run(tmp_path, capsys)
    assert code == 0
    assert "Top risk contributor: HIGH" in out


@pytest.mark.parametrize("extra", [
    ["--weights", "LOW=1"],
    ["--weights", "LOW=1,NOPE=1"],
    ["--weights", "LOW=x,MID=1"],
    ["--weights", "LOW=1,LOW=2"],
    ["--weights", "LOW=1,MID=-1"],
    ["--lookback", "5"],
    ["--confidence", "1.0"],
])
def test_riskbudget_rejects_bad_input(tmp_path, capsys, extra):
    code, out = _run(tmp_path, capsys, *extra)
    assert code == 1
    assert "Error" in out
