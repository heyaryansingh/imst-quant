"""DSR from returns must not use an annualized Sharpe in the per-period stderr."""

import numpy as np

from imst_quant.utils.deflated_sharpe import (
    deflated_sharpe_ratio_from_returns,
    probabilistic_sharpe_ratio,
)


def _series(n, sharpe, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    return 0.01 * (x + sharpe)  # exact per-period Sharpe


def test_annualization_does_not_change_probability():
    r = _series(60, 0.1)
    daily = deflated_sharpe_ratio_from_returns(r, n_trials=1, periods_per_year=1)
    annual = deflated_sharpe_ratio_from_returns(r, n_trials=1, periods_per_year=252)
    assert abs(daily["deflated_sharpe_ratio"] - annual["deflated_sharpe_ratio"]) < 1e-12
    # 60 bars at SR 0.1 is weak evidence; the old code reported ~1.0 here.
    assert 0.6 < annual["deflated_sharpe_ratio"] < 0.9
    assert abs(annual["sharpe_ratio"] - 0.1 * np.sqrt(252)) < 1e-9


def test_matches_psr_with_one_trial():
    r = _series(500, 0.05)
    res = deflated_sharpe_ratio_from_returns(r, n_trials=1, periods_per_year=252)
    from scipy import stats

    psr = probabilistic_sharpe_ratio(
        0.05, 0.0, 500, float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    )
    assert abs(res["deflated_sharpe_ratio"] - psr) < 1e-9


def test_many_trials_benchmark_is_on_null_sampling_scale():
    # Expected max of 100 zero-skill trials ~ 2.5 sigma of the estimator,
    # not 2.5 * sqrt(252) annualized Sharpe (which made DSR ~0 for everything).
    r = _series(1000, 0.15)
    res = deflated_sharpe_ratio_from_returns(r, n_trials=100, periods_per_year=1)
    assert 0.05 < res["expected_max_sharpe"] < 0.12
    assert res["deflated_sharpe_ratio"] > 0.95


def test_nans_dropped():
    r = _series(100, 0.1)
    with_nan = np.concatenate([r, [np.nan, np.inf]])
    a = deflated_sharpe_ratio_from_returns(r, n_trials=3)
    b = deflated_sharpe_ratio_from_returns(with_nan, n_trials=3)
    assert a["n_observations"] == b["n_observations"] == 100
    assert abs(a["deflated_sharpe_ratio"] - b["deflated_sharpe_ratio"]) < 1e-12


def test_min_track_record_length_inverts_psr():
    from imst_quant.utils.deflated_sharpe import min_track_record_length

    sr, skew, kurt = 0.08, -0.5, 6.0
    n = min_track_record_length(sr, 0.0, skew, kurt, confidence=0.95)
    assert abs(probabilistic_sharpe_ratio(sr, 0.0, int(np.ceil(n)), skew, kurt) - 0.95) < 0.01
    assert probabilistic_sharpe_ratio(sr, 0.0, int(np.ceil(n)), skew, kurt) >= 0.95
    assert probabilistic_sharpe_ratio(sr, 0.0, int(np.floor(n)) - 1, skew, kurt) < 0.95
    assert min_track_record_length(-0.1) == float("inf")


def test_cli_reports_min_track_record(tmp_path, capsys):
    import json

    import polars as pl

    from imst_quant.cli import cmd_deflated_sharpe, create_parser

    path = tmp_path / "returns.parquet"
    pl.DataFrame({"returns": _series(120, 0.1)}).write_parquet(path)
    args = create_parser().parse_args(
        ["deflated-sharpe", "--returns", str(path), "--n-trials", "5", "--json"]
    )
    assert cmd_deflated_sharpe(args) == 0
    out = json.loads(capsys.readouterr().out)
    # Not enough history yet for 5 trials at SR 0.1/bar.
    assert out["deflated_sharpe_ratio"] < 0.95
    assert out["min_track_record_length"] > 120
