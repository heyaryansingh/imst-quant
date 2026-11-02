import pandas as pd
from imst_quant.utils.alpha_metrics import AlphaMetrics


def test_brinson_total_is_active_return_once():
    r = pd.Series([0.0, 0.0])
    m = AlphaMetrics(r, r)
    pw = pd.DataFrame({"a": [.6], "b": [.4]})
    bw = pd.DataFrame({"a": [.5], "b": [.5]})
    sr = pd.DataFrame({"a": [.1], "b": [0.0]})
    d = m.brinson_attribution(pw, bw, sr)
    assert abs(d.total_alpha - 0.01) < 1e-12
    assert abs(d.selection_alpha + d.timing_alpha + d.interaction_alpha - d.total_alpha) < 1e-12
