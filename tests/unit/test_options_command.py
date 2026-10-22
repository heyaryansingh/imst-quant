"""Tests for the `options` CLI command."""

import json

import numpy as np
import pytest

from imst_quant.cli import cmd_options, create_parser


def _run(capsys, *argv):
    args = create_parser().parse_args(["options", *argv])
    code = cmd_options(args)
    return code, capsys.readouterr().out


def test_both_legs_satisfy_parity(capsys):
    code, out = _run(capsys, "--spot", "100", "--strike", "100", "--days", "73",
                     "--vol", "0.2", "--rate", "0.05", "--json")
    assert code == 0
    p = json.loads(out)
    call, put = p["options"]["call"], p["options"]["put"]
    t = 73 / 365
    assert call["price"] - put["price"] == pytest.approx(100 - 100 * np.exp(-0.05 * t))
    assert call["delta"] - put["delta"] == pytest.approx(1.0)
    assert call["gamma"] == pytest.approx(put["gamma"])


def test_implied_vol_recovers_input(capsys):
    _, out = _run(capsys, "--spot", "50", "--strike", "55", "--days", "90",
                  "--vol", "0.4", "--type", "call", "--json")
    premium = json.loads(out)["options"]["call"]["price"]
    code, out = _run(capsys, "--spot", "50", "--strike", "55", "--days", "90",
                     "--market-price", str(premium), "--type", "call", "--json")
    assert code == 0
    p = json.loads(out)
    assert p["implied"] is True
    assert p["volatility"] == pytest.approx(0.4, abs=1e-4)


def test_text_output(capsys):
    code, out = _run(capsys, "--spot", "100", "--strike", "90", "--days", "30", "--vol", "0.3")
    assert code == 0
    assert "Black-Scholes" in out and "Delta" in out and "Put" in out


@pytest.mark.parametrize("extra", [
    ["--vol", "0"],
    ["--vol", "0.2", "--days", "-1"],
    ["--market-price", "3"],  # needs a single --type
    ["--market-price", "3", "--type", "call", "--div", "0.02"],
    ["--market-price", "1000", "--type", "call"],  # above spot, no IV
])
def test_rejects_bad_input(capsys, extra):
    base = ["--spot", "100", "--strike", "100", "--days", "30"]
    if "--days" in extra:
        base = base[:4]
    code, _ = _run(capsys, *base, *extra)
    assert code == 1
