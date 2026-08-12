"""Tests for the Gini coefficient used by the concentration metrics.

The Lorenz curve behind the Gini coefficient starts at the origin. Integrating
it from the first position instead dropped 1/n of the area, so an equal-weight
book scored 1/n^2 rather than 0 and every portfolio looked slightly more
unequal than it was.
"""

import numpy as np
import polars as pl
import pytest

from imst_quant.utils.concentration_metrics import gini_coefficient


def _reference_gini(values):
    """Mean absolute difference definition, independent of the implementation."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    pairwise = np.abs(values[:, None] - values[None, :]).sum()
    return pairwise / (2 * n**2 * values.mean())


@pytest.mark.parametrize("n", [1, 2, 4, 10])
def test_equal_weights_score_zero(n):
    weights = pl.Series("weight", [1.0 / n] * n)

    assert gini_coefficient(weights) == pytest.approx(0.0, abs=1e-12)


def test_matches_the_mean_absolute_difference_definition():
    weights = [0.7, 0.2, 0.1]

    assert gini_coefficient(pl.Series("weight", weights)) == pytest.approx(
        _reference_gini(weights)
    )


@pytest.mark.parametrize(
    "weights",
    [
        [0.4, 0.3, 0.2, 0.1],
        [0.5, 0.25, 0.15, 0.10],
        [0.9, 0.05, 0.03, 0.02],
        [0.34, 0.33, 0.33],
    ],
)
def test_matches_reference_on_varied_books(weights):
    assert gini_coefficient(pl.Series("weight", weights)) == pytest.approx(
        _reference_gini(weights)
    )


def test_concentrated_book_scores_higher_than_a_diversified_one():
    concentrated = pl.Series("weight", [0.9, 0.05, 0.03, 0.02])
    diversified = pl.Series("weight", [0.3, 0.3, 0.2, 0.2])

    assert gini_coefficient(concentrated) > gini_coefficient(diversified)


def test_approaches_one_as_a_single_name_dominates():
    weights = pl.Series("weight", [1.0] + [0.0] * 999)

    # Population Gini tops out at (n-1)/n for a single non-zero position.
    assert gini_coefficient(weights) == pytest.approx(0.999)


def test_is_scale_invariant():
    small = pl.Series("weight", [0.4, 0.3, 0.2, 0.1])
    large = pl.Series("weight", [4000.0, 3000.0, 2000.0, 1000.0])

    assert gini_coefficient(small) == pytest.approx(gini_coefficient(large))


def test_order_does_not_matter():
    ascending = pl.Series("weight", [0.1, 0.2, 0.3, 0.4])
    shuffled = pl.Series("weight", [0.3, 0.1, 0.4, 0.2])

    assert gini_coefficient(ascending) == pytest.approx(gini_coefficient(shuffled))


def test_dataframe_input_uses_the_named_column():
    frame = pl.DataFrame({"symbol": ["A", "B"], "w": [0.75, 0.25]})

    assert gini_coefficient(frame, weight_col="w") == pytest.approx(
        gini_coefficient(pl.Series("weight", [0.75, 0.25]))
    )


def test_empty_and_all_zero_inputs_are_zero():
    assert gini_coefficient(pl.Series("weight", [], dtype=pl.Float64)) == 0.0
    assert gini_coefficient(pl.Series("weight", [0.0, 0.0])) == 0.0
