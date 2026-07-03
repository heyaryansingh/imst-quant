"""Tests for the position pyramiding (scale-in) module.

Tests cover:
- Pyramid level generation for long and short directions
- Decreasing add-on sizing
- Trigger detection as price moves favorably
- Total position size accounting
"""

import pytest

from imst_quant.trading.position_pyramiding import (
    generate_pyramid_levels,
    next_pyramid_trigger,
    total_position_size,
)


def test_generate_pyramid_levels_long_spacing_and_sizing():
    levels = generate_pyramid_levels(
        entry_price=100.0, atr=2.0, max_adds=3, atr_multiplier=1.5, size_decay=0.5
    )
    assert len(levels) == 3
    assert levels[0].trigger_price == pytest.approx(103.0)
    assert levels[1].trigger_price == pytest.approx(106.0)
    assert levels[2].trigger_price == pytest.approx(109.0)

    # Sizes decay: 0.5, 0.25, 0.125
    assert levels[0].size_fraction == pytest.approx(0.5)
    assert levels[1].size_fraction == pytest.approx(0.25)
    assert levels[2].size_fraction == pytest.approx(0.125)

    # Cumulative fractions are monotonically increasing
    assert levels[0].cumulative_size_fraction < levels[1].cumulative_size_fraction
    assert levels[1].cumulative_size_fraction < levels[2].cumulative_size_fraction


def test_generate_pyramid_levels_short_direction():
    levels = generate_pyramid_levels(
        entry_price=100.0, atr=2.0, max_adds=2, atr_multiplier=1.0, direction="short"
    )
    assert levels[0].trigger_price == pytest.approx(98.0)
    assert levels[1].trigger_price == pytest.approx(96.0)


def test_generate_pyramid_levels_invalid_args():
    with pytest.raises(ValueError):
        generate_pyramid_levels(100.0, atr=2.0, max_adds=0)
    with pytest.raises(ValueError):
        generate_pyramid_levels(100.0, atr=-1.0)
    with pytest.raises(ValueError):
        generate_pyramid_levels(100.0, atr=2.0, size_decay=1.5)
    with pytest.raises(ValueError):
        generate_pyramid_levels(100.0, atr=2.0, direction="sideways")


def test_next_pyramid_trigger_long():
    levels = generate_pyramid_levels(100.0, atr=2.0, max_adds=3, atr_multiplier=1.0)
    # levels trigger at 102, 104, 106
    triggered = next_pyramid_trigger(103.5, levels, executed_levels=set())
    assert triggered is not None
    assert triggered.level == 1

    triggered_none = next_pyramid_trigger(101.0, levels, executed_levels=set())
    assert triggered_none is None


def test_next_pyramid_trigger_skips_executed():
    levels = generate_pyramid_levels(100.0, atr=2.0, max_adds=3, atr_multiplier=1.0)
    triggered = next_pyramid_trigger(107.0, levels, executed_levels={1})
    assert triggered is not None
    assert triggered.level == 2


def test_total_position_size():
    levels = generate_pyramid_levels(100.0, atr=2.0, max_adds=2, size_decay=0.5)
    size_no_adds = total_position_size(1000.0, levels, executed_levels=set())
    assert size_no_adds == pytest.approx(1000.0)

    size_one_add = total_position_size(1000.0, levels, executed_levels={1})
    assert size_one_add == pytest.approx(1500.0)

    size_both_adds = total_position_size(1000.0, levels, executed_levels={1, 2})
    assert size_both_adds == pytest.approx(1750.0)
