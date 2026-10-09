"""Tests for recovery_speed milestone timing and pattern classification."""

import polars as pl

from imst_quant.utils.recovery_speed import RecoveryPattern, calculate_recovery_rate


def test_milestones_count_bars_after_trough():
    # Trough wealth 0.8; 0.8*1.05^3 >= 0.9 (50%), 0.8*1.05^5 >= 0.98 (90%).
    # Re-applying the -20% trough bar used to leave both milestones as None.
    r = pl.Series([0.0, -0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.02])
    (m,) = calculate_recovery_rate(r)
    assert (m.time_to_50pct, m.time_to_90pct) == (3, 5)


def test_slow_recovery_is_l_shaped():
    # Recovery takes far more than twice the fall; L-shaped used to be
    # unreachable because the U-shaped branch caught it first.
    r = pl.Series([0.0, -0.1] + [0.012] * 9 + [0.0])
    (m,) = calculate_recovery_rate(r)
    assert m.pattern == RecoveryPattern.L_SHAPED
