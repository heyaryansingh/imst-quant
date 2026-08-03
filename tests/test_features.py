"""Tests for feature engineering (Phase 6)."""

import polars as pl
import pytest

from imst_quant.features import build_daily_features
from imst_quant.features.builder import _returns_and_vol


def _market(closes):
    """Build a market frame with one row per ascending date."""
    dates = [f"2024-01-{i + 1:02d}" for i in range(len(closes))]
    return pl.DataFrame({"date": dates, "close": closes})


class TestReturnsAndVol:
    """Tests for the return/volatility feature primitives."""

    def test_too_few_rows_returns_zeros(self):
        assert _returns_and_vol(_market([100.0])) == (0.0, 0.0, 0.0)

    def test_missing_close_column_returns_zeros(self):
        df = pl.DataFrame({"date": ["2024-01-01", "2024-01-02"], "open": [1.0, 2.0]})
        assert _returns_and_vol(df) == (0.0, 0.0, 0.0)

    def test_ret_1d_is_last_bar_over_previous(self):
        ret_1d, _, _ = _returns_and_vol(_market([100.0, 110.0]))
        assert ret_1d == pytest.approx(0.10)

    def test_ret_5d_spans_five_trading_days(self):
        # Six bars: the 5-day return must compare bar 6 against bar 1,
        # not bar 6 against bar 2.
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        _, ret_5d, _ = _returns_and_vol(_market(closes))
        assert ret_5d == pytest.approx(110.0 / 100.0 - 1)

    def test_ret_5d_falls_back_to_1d_without_six_bars(self):
        closes = [100.0, 101.0, 102.0, 103.0, 110.0]
        ret_1d, ret_5d, _ = _returns_and_vol(_market(closes))
        assert ret_5d == ret_1d

    def test_vol_is_population_std_of_returns(self):
        # Returns are +10%, -10%, +10%; population std about a mean of 1/30.
        _, _, vol = _returns_and_vol(_market([100.0, 110.0, 99.0, 108.9]))
        assert vol == pytest.approx(0.0942809041582, abs=1e-9)

    def test_constant_prices_give_zero_vol(self):
        _, _, vol = _returns_and_vol(_market([100.0] * 10))
        assert vol == pytest.approx(0.0)

    def test_zero_close_is_skipped_not_divided_by(self):
        # A zero close is bad data; it must not raise or produce -100%.
        ret_1d, _, vol = _returns_and_vol(_market([100.0, 0.0, 105.0, 110.0]))
        assert ret_1d == pytest.approx(110.0 / 105.0 - 1)
        assert vol == pytest.approx(0.0, abs=1e-9)

    def test_unsorted_input_is_sorted_by_date(self):
        df = pl.DataFrame(
            {
                "date": ["2024-01-03", "2024-01-01", "2024-01-02"],
                "close": [120.0, 100.0, 110.0],
            }
        )
        ret_1d, _, _ = _returns_and_vol(df)
        assert ret_1d == pytest.approx(120.0 / 110.0 - 1)


def test_build_daily_features(tmp_path):
    """FEAT-01: Feature builder produces gold parquet."""
    bronze = tmp_path / "market" / "date=2024-01-14"
    bronze.mkdir(parents=True)
    (tmp_path / "market" / "date=2024-01-10").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-11").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-12").mkdir(parents=True, exist_ok=True)
    (tmp_path / "market" / "date=2024-01-13").mkdir(parents=True, exist_ok=True)
    for d, c in [("2024-01-10", 100.0), ("2024-01-11", 101.0), ("2024-01-12", 102.0),
                 ("2024-01-13", 101.5), ("2024-01-14", 103.0)]:
        p = tmp_path / "market" / f"date={d}"
        p.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({
            "ticker": ["AAPL"], "date": [d], "close": [c],
        }).write_parquet(p / "data.parquet")

    sent = tmp_path / "sentiment.parquet"
    pl.DataFrame({
        "date": ["2024-01-15"],
        "asset_id": ["AAPL"],
        "sentiment_index": [0.2],
        "post_count": [10],
    }).write_parquet(sent)

    out = tmp_path / "gold" / "features.parquet"
    df = build_daily_features(
        tmp_path, sent, out,
        assets=["AAPL"],
        start_date="2024-01-15",
        end_date="2024-01-15",
    )
    assert len(df) >= 1
    assert "return_1d" in df.columns
    assert "sentiment_index" in df.columns
