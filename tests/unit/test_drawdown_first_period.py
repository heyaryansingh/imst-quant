"""A loss in the first period is a drawdown from initial capital."""

import polars as pl
import pytest

from imst_quant.utils.drawdown_analysis import (
    drawdown_statistics,
    identify_drawdown_periods,
)


def test_first_period_loss_counts():
    stats = drawdown_statistics(pl.Series([-0.10, 0.05]))
    assert stats["max_drawdown"] == pytest.approx(0.10)


def test_first_period_loss_period_recovers():
    periods = identify_drawdown_periods(pl.Series([-0.10, 0.05, 0.10]), min_drawdown=0.05)
    assert len(periods) == 1
    p = periods[0]
    assert p.start_idx == 0 and p.peak_value == 1.0
    assert p.is_recovered and p.end_idx == 2


def test_sibling_max_drawdown_helpers_count_first_period_loss():
    import numpy as np

    from imst_quant.utils.portfolio_snapshot import _calculate_max_drawdown

    assert _calculate_max_drawdown(np.array([-0.10, 0.05])) == pytest.approx(0.10)
