"""Tests for the `impact` CLI command."""

import datetime as dt
import json

import polars as pl
import pytest

from imst_quant.cli import cmd_impact, create_parser


def _run(capsys, *argv):
    args = create_parser().parse_args(["impact", *argv])
    code = cmd_impact(args)
    return code, capsys.readouterr().out


def test_flags_only_json(capsys):
    code, out = _run(
        capsys, "--shares", "-50000", "--price", "100", "--adv", "1000000",
        "--volatility", "0.02", "--horizons", "0.25,2,20", "--json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["shares"] == 50000  # sells cost like buys
    assert payload["participation_rate"] == pytest.approx(0.05)
    costs = [r["cost_bps"] for r in payload["almgren_chriss"]]
    risks = [r["timing_risk_bps"] for r in payload["almgren_chriss"]]
    assert costs == sorted(costs, reverse=True)
    assert risks == sorted(risks)
    # Neither extreme wins once timing risk is priced in.
    assert payload["best_horizon_days"] == 2.0
    assert payload["inputs_source"] == {"adv": "flag", "volatility": "flag", "price": "flag"}


def test_zero_risk_aversion_picks_longest_horizon(capsys):
    code, out = _run(
        capsys, "--shares", "1000", "--price", "10", "--adv", "100000",
        "--horizons", "1,5", "--risk-aversion", "0", "--json",
    )
    assert code == 0
    payload = json.loads(out)
    assert payload["best_horizon_days"] == 5.0
    assert payload["inputs_source"]["volatility"] == "default"


def test_derives_inputs_from_data_tail(tmp_path, capsys):
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(5)]
    frame = pl.DataFrame({
        "date": list(reversed(dates)),  # unsorted on disk
        "volume": [500.0, 400.0, float("nan"), 200.0, 1e9],
        "returns": [0.01, -0.01, 0.01, -0.01, 0.5],
        "close": [15.0, 14.0, 13.0, 12.0, 11.0],
    })
    path = tmp_path / "px.parquet"
    frame.write_parquet(path)
    code, out = _run(
        capsys, "--shares", "100", "--data", str(path), "--lookback", "3", "--json",
    )
    assert code == 0
    payload = json.loads(out)
    # Sorted by date, the last 3 finite volumes are 200, 400, 500 (NaN dropped).
    assert payload["adv"] == pytest.approx((200 + 400 + 500) / 3)
    assert payload["price"] == 15.0
    assert payload["daily_volatility"] == pytest.approx(
        pl.Series([0.01, -0.01, 0.01]).std()
    )
    assert set(payload["inputs_source"].values()) == {"data"}


@pytest.mark.parametrize("argv", [
    ["--shares", "100"],  # nothing to derive ADV/price from
    ["--shares", "0", "--price", "1", "--adv", "10"],
    ["--shares", "100", "--price", "1", "--adv", "0"],
    ["--shares", "100", "--price", "1", "--adv", "10", "--horizons", "1,-2"],
    ["--shares", "100", "--price", "1", "--adv", "10", "--horizons", "x"],
    ["--shares", "100", "--data", "missing.parquet"],
])
def test_rejects_bad_input(capsys, argv):
    code, out = _run(capsys, *argv)
    assert code == 1
    assert out.startswith("Error")


def test_text_report_warns_on_large_order(capsys):
    code, out = _run(capsys, "--shares", "300000", "--price", "50", "--adv", "1000000")
    assert code == 0
    assert "<- best" in out
    assert "exceeds 20% of ADV" in out
