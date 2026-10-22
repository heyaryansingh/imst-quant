"""Tests for trade clustering behavioral helpers."""

import numpy as np
import pandas as pd

from imst_quant.utils.trade_clustering import analyze_win_loss_clustering


def test_win_loss_streaks_skip_missing_pnl():
    trades = pd.DataFrame({"pnl": [1.0, np.nan, 2.0, -1.0, None, -2.0, np.inf]})
    result = analyze_win_loss_clustering(trades)
    assert result["max_win_streak"] == 2
    assert result["max_loss_streak"] == 2
    assert result["total_streaks"] == 2
