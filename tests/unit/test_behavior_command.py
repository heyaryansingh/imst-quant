"""Tests for the `behavior` CLI command."""

import json
import sys
from datetime import datetime, timedelta, timezone

import pytest
import structlog

from imst_quant.cli import cmd_behavior, create_parser
from imst_quant.utils.trade_journal import TradeEntry, TradeJournal

T0 = datetime(2026, 1, 5, 14, 0, tzinfo=timezone.utc)


def _journal(tmp_path):
    """Six trades: #3 is a revenge trade (30 min after #2's loss) that also
    doubles size; #4 enters a day later at normal size."""
    path = tmp_path / "journal.json"
    journal = TradeJournal(path)
    plan = [
        # (entry offset h, hold h, qty, exit price)
        (0, 1, 100, 101.0),
        (24, 1, 100, 98.0),  # loss, exits at 25h
        (25.5, 1, 200, 97.0),  # revenge + size-up, loss
        (49.5, 1, 100, 102.0),
        (72, 1, 100, 101.0),
        (96, 1, 100, 99.5),
    ]
    for i, (start, hold, qty, exit_price) in enumerate(plan):
        tid = journal.log_entry(
            TradeEntry(
                trade_id=f"T{i}",
                symbol="AAPL",
                direction="long",
                entry_price=100.0,
                quantity=qty,
                entry_time=T0 + timedelta(hours=start),
            )
        )
        journal.close_trade(
            tid, exit_price, exit_time=T0 + timedelta(hours=start + hold)
        )
    journal.log_entry(
        TradeEntry(symbol="MSFT", direction="long", entry_price=10.0, quantity=1)
    )
    journal.save()
    return path


@pytest.fixture(autouse=True)
def _logs_to_stderr():
    structlog.configure(logger_factory=structlog.PrintLoggerFactory(sys.stderr))
    yield
    structlog.reset_defaults()


def _run(capsys, *argv):
    args = create_parser().parse_args(["behavior", *argv])
    code = cmd_behavior(args)
    return code, capsys.readouterr().out


def test_behavior_flags_revenge_and_size_up(tmp_path, capsys):
    code, out = _run(capsys, "--journal", str(_journal(tmp_path)), "--json")
    assert code == 0
    payload = json.loads(out)
    assert payload["closed_trades"] == 6  # open MSFT trade ignored
    assert payload["revenge_trade_ids"] == ["T2"]
    groups = payload["groups"]
    assert groups["revenge"]["trades"] == 1
    assert groups["revenge"]["total_pnl"] == pytest.approx(-600.0)
    assert groups["size_up_after_loss"]["trades"] == 1
    assert payload["clean"]["trades"] == 5
    assert payload["streaks"]["max_loss_streak"] == 2


def test_behavior_window_controls_revenge(tmp_path, capsys):
    code, out = _run(
        capsys, "--journal", str(_journal(tmp_path)), "--json", "--window-hours", "0.25"
    )
    assert code == 0
    assert json.loads(out)["revenge_trade_ids"] == []


def test_behavior_text_output(tmp_path, capsys):
    code, out = _run(capsys, "--journal", str(_journal(tmp_path)))
    assert code == 0
    assert "revenge" in out and "Longest win/loss streak: 2/2" in out


def test_behavior_rejects_missing_journal_and_bad_window(tmp_path, capsys):
    assert _run(capsys, "--journal", str(tmp_path / "x.json"))[0] == 1
    path = _journal(tmp_path)
    code, out = _run(capsys, "--journal", str(path), "--window-hours", "0")
    assert code == 1 and out.startswith("Error")
