import math

import numpy as np
import pandas as pd

from imst_quant.utils.sentiment_strength import SentimentStrengthAnalyzer


def test_non_finite_polarity_rows_are_dropped():
    pol = [0.3] * 19 + [np.nan, np.inf]
    df = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=len(pol)),
        "asset_id": "A",
        "polarity": pol,
        "volume": 20,
        "subjectivity": 0.2,
    })
    out = SentimentStrengthAnalyzer().analyze_sentiment_data(df, "A")
    assert all(math.isfinite(v) for v in out.values())
    assert out["momentum"] == 0.0
    assert out["strength_score"] > 0
