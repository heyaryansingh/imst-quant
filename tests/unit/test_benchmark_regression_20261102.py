import polars as pl
from imst_quant.utils.benchmark import BenchmarkAnalyzer


def test_null_pairs_dropped_together():
    a = BenchmarkAnalyzer(pl.Series([.01, None, .03, .02, -.01]), pl.Series([.01, .02, .03, .02, -.01]))
    assert list(a.benchmark_returns) == [.01, .03, .02, -.01]
    assert abs(a.beta - 1.0) < 1e-9
