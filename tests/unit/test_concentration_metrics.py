"""Tests for concentration metrics.

`gini_coefficient` integrated with `np.trapz`, which numpy 2.0 removed, so on
any numpy>=2 install every concentration metric raised AttributeError. These
tests pin the values so the shim cannot silently regress.
"""

import polars as pl
import pytest

from imst_quant.utils.concentration_metrics import (
    calculate_all_concentration,
    effective_n,
    gini_coefficient,
    herfindahl_index,
)


def test_gini_is_zero_for_an_equal_weighted_book():
    assert gini_coefficient(pl.Series([0.25, 0.25, 0.25, 0.25])) == pytest.approx(0.0)


def test_gini_rises_with_inequality():
    equal = gini_coefficient(pl.Series([0.25, 0.25, 0.25, 0.25]))
    skewed = gini_coefficient(pl.Series([0.85, 0.05, 0.05, 0.05]))

    assert skewed > equal
    assert 0.0 <= skewed <= 1.0


def test_gini_ignores_scale():
    """Gini is scale-invariant: doubling every weight changes nothing."""
    weights = [0.4, 0.3, 0.2, 0.1]

    assert gini_coefficient(pl.Series(weights)) == pytest.approx(
        gini_coefficient(pl.Series([w * 2 for w in weights]))
    )


def test_gini_matches_the_pairwise_difference_definition():
    """Cross-check the Lorenz integration against the textbook Gini formula."""
    weights = [0.85, 0.05, 0.05, 0.05]
    n = len(weights)
    mean = sum(weights) / n
    expected = sum(
        abs(a - b) for a in weights for b in weights
    ) / (2 * n**2 * mean)

    assert gini_coefficient(pl.Series(weights)) == pytest.approx(expected)


def test_gini_handles_an_empty_book():
    assert gini_coefficient(pl.Series([], dtype=pl.Float64)) == 0.0


def test_hhi_and_effective_n_agree():
    weights = pl.Series([0.5, 0.3, 0.2])

    assert herfindahl_index(weights) == pytest.approx(0.38)
    assert effective_n(weights) == pytest.approx(1 / 0.38)


def test_calculate_all_concentration_returns_every_metric():
    metrics = calculate_all_concentration(pl.Series([0.25, 0.25, 0.25, 0.25]), top_n=2)

    assert metrics["hhi"] == pytest.approx(0.25)
    assert metrics["effective_n"] == pytest.approx(4.0)
    assert metrics["top_2_concentration"] == pytest.approx(0.5)
    assert metrics["gini"] == pytest.approx(0.0)
    assert metrics["normalized_entropy"] == pytest.approx(1.0)
    assert metrics["n_positions"] == 4
