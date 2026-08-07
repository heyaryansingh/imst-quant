"""Tests for technical indicators on degenerate (zero-range) price bars.

Flat bars are common in real data — halted symbols, illiquid small caps,
after-hours candles — and every indicator below used to divide by a zero
range, emitting NaN that silently propagated into downstream features.
"""

import math

import polars as pl
import pytest

from imst_quant.utils.technical_indicators import (
    adx,
    bollinger_bands,
    cci,
    rsi,
    stochastic_oscillator,
    williams_r,
)


@pytest.fixture
def flat() -> pl.DataFrame:
    """Ten bars with identical high, low and close."""
    return pl.DataFrame({
        "high": [100.0] * 10,
        "low": [100.0] * 10,
        "close": [100.0] * 10,
        "volume": [1000.0] * 10,
    })


@pytest.fixture
def rising() -> pl.DataFrame:
    """Ten strictly rising bars with a non-zero range."""
    return pl.DataFrame({
        "high": [100.0 + i for i in range(10)],
        "low": [99.0 + i for i in range(10)],
        "close": [99.5 + i for i in range(10)],
        "volume": [1000.0] * 10,
    })


def _settled(df: pl.DataFrame, column: str) -> list[float]:
    """Return the non-null values of a column, i.e. past the warmup window."""
    return [v for v in df[column].to_list() if v is not None]


class TestFlatBarsAreNeutral:
    """A zero-width range must yield the neutral reading, never NaN."""

    def test_stochastic_is_midpoint(self, flat):
        assert _settled(stochastic_oscillator(flat, k_period=3), "stoch_k") == [50.0] * 8

    def test_williams_r_is_midpoint(self, flat):
        assert _settled(williams_r(flat, period=3), "williams_r") == [-50.0] * 8

    def test_cci_is_zero(self, flat):
        assert _settled(cci(flat, period=3), "cci") == [0.0] * 8

    def test_rsi_is_neutral(self, flat):
        assert _settled(rsi(flat, period=3), "rsi") == [50.0] * 10

    def test_adx_has_no_directional_movement(self, flat):
        result = adx(flat, window=3)
        assert _settled(result, "plus_di") == [0.0] * 10
        assert _settled(result, "minus_di") == [0.0] * 10
        assert _settled(result, "adx") == [0.0] * 10

    def test_bollinger_bands_collapse_to_midpoint(self, flat):
        result = bollinger_bands(flat, window=3)
        assert _settled(result, "bb_width") == [0.0] * 8
        assert _settled(result, "bb_percent") == [0.5] * 8

    def test_no_indicator_emits_nan(self, flat):
        frames = {
            "stoch_k": stochastic_oscillator(flat, k_period=3),
            "williams_r": williams_r(flat, period=3),
            "cci": cci(flat, period=3),
            "rsi": rsi(flat, period=3),
            "adx": adx(flat, window=3),
            "bb_percent": bollinger_bands(flat, window=3),
        }
        for column, frame in frames.items():
            assert not any(math.isnan(v) for v in _settled(frame, column)), column


class TestWarmupAndRange:
    """The zero-range guard must not disturb ordinary bars."""

    def test_warmup_rows_stay_null(self, rising):
        # Rolling windows have no value before k_period bars; those rows must
        # remain null rather than being filled with the neutral fallback.
        stoch_k = stochastic_oscillator(rising, k_period=3)["stoch_k"].to_list()
        assert stoch_k[:2] == [None, None]
        assert stoch_k[2] is not None

    def test_rsi_saturates_at_100_when_never_losing(self, rising):
        assert rsi(rising, period=3)["rsi"].to_list()[-1] == pytest.approx(100.0)

    def test_rsi_saturates_at_0_when_never_gaining(self):
        falling = pl.DataFrame({"close": [110.0 - i for i in range(10)]})
        assert rsi(falling, period=3)["rsi"].to_list()[-1] == pytest.approx(0.0)

    def test_stochastic_locates_close_within_the_window_range(self, rising):
        # Final 3 bars: highs 107..109, lows 106..108, close 108.5.
        assert stochastic_oscillator(rising, k_period=3)["stoch_k"].to_list()[-1] == (
            pytest.approx(100 * (108.5 - 106.0) / (109.0 - 106.0))
        )

    def test_williams_r_mirrors_stochastic(self, rising):
        stoch_k = stochastic_oscillator(rising, k_period=3)["stoch_k"].to_list()[-1]
        will_r = williams_r(rising, period=3)["williams_r"].to_list()[-1]
        assert will_r == pytest.approx(stoch_k - 100)
