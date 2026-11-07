import math

import pytest

from imst_quant.utils.strategy_comparison import (
    calculate_relative_performance,
    find_best_strategy,
    rank_strategies,
)


def test_drawdown_tiebreak_prefers_shallower_drawdown_either_sign():
    neg = {"LSTM": {"sharpe": 1.5, "max_drawdown": -0.12}, "CNN": {"sharpe": 1.5, "max_drawdown": -0.10}}
    pos = {"LSTM": {"sharpe": 1.5, "max_drawdown": 0.12}, "CNN": {"sharpe": 1.5, "max_drawdown": 0.10}}
    assert find_best_strategy(neg) == "CNN"
    assert find_best_strategy(pos) == "CNN"


def test_nan_primary_metric_is_skipped_not_crash():
    res = {"bad": {"sharpe": math.nan}, "good": {"sharpe": 0.8}}
    assert find_best_strategy(res) == "good"
    with pytest.raises(ValueError):
        find_best_strategy({"bad": {"sharpe": math.nan}})


def test_relative_performance_negative_benchmark_keeps_sign():
    res = {"A": {"total_return": 0.05}, "B": {"total_return": -0.10}, "C": {"total_return": -0.20}}
    out = calculate_relative_performance(res, "B")
    assert out["A"] == pytest.approx(1.5)
    assert out["C"] == pytest.approx(-1.0)
    assert out["B"] == 0.0


def test_rank_drops_nan():
    res = {"a": {"sharpe": math.nan}, "b": {"sharpe": 1.0}, "c": {"sharpe": 2.0}}
    assert rank_strategies(res) == [("c", 2.0), ("b", 1.0)]
