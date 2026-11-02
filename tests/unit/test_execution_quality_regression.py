from imst_quant.utils.execution_quality import compare_venues


def test_compare_venues_empty_returns_empty_frame():
    df = compare_venues([])
    assert df.empty and "quality_score" in df.columns
