"""Tests for streak_analysis expected-max-streak and ruin probability."""

import numpy as np
import polars as pl

from imst_quant.utils.streak_analysis import analyze_streaks, calculate_gambler_ruin_prob


def test_expected_max_streak_is_positive_and_sane():
    # Fair coin, n=100: expected longest run ~ log2(50) ~ 5.6. The old formula
    # divided by log(p) < 0 and returned about -4.3.
    rng = np.random.default_rng(0)
    stats = analyze_streaks(pl.Series(rng.normal(size=100)))
    assert 5.0 < stats.expected_max_streak < 7.5


def test_ruin_matches_classic_formula_for_even_payoff():
    res = calculate_gambler_ruin_prob(0.6, 1.0, 1.0, 10)
    assert abs(res["ruin_probability"] - (0.4 / 0.6) ** 10) < 1e-9


def test_positive_edge_below_half_win_rate_is_not_certain_ruin():
    # 40% winners paying 2:1 is a positive-edge strategy; the old q/p > 1 power
    # clipped ruin to 100%. Monte Carlo puts it near 15%.
    res = calculate_gambler_ruin_prob(0.4, 2.0, 1.0, 10)
    assert res["is_positive_expectancy"]
    assert 0.10 < res["ruin_probability"] < 0.20
