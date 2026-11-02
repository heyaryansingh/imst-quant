from datetime import date

from imst_quant.utils.tax_lots import LotSale


def _sale(acq, disp):
    return LotSale("X", 1, 10, 12, acq, disp, (disp - acq).days, 1)


def test_sale_on_anniversary_is_short_term():
    assert not _sale(date(2025, 3, 1), date(2026, 3, 1)).is_long_term
    assert _sale(date(2025, 3, 1), date(2026, 3, 2)).is_long_term


def test_leap_year_365_days_is_short_term():
    # 2023-03-01 -> 2024-02-29 is 365 days but not yet a full year
    assert not _sale(date(2023, 3, 1), date(2024, 2, 29)).is_long_term
    assert _sale(date(2024, 2, 29), date(2025, 3, 1)).is_long_term
    assert not _sale(date(2024, 2, 29), date(2025, 2, 28)).is_long_term
