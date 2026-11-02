"""Las métricas dan el resultado correcto sobre curvas de capital sintéticas de resultado conocido (Paula)."""
import math

import numpy as np
import pandas as pd
import pytest

from performance.metrics import (cagr, calmar_ratio, deflated_sharpe, expected_max_sharpe, max_drawdown,
                                 probabilistic_sharpe, profit_factor, sharpe_ratio, sortino_ratio, win_rate)
from performance.rolling import rolling_sharpe

A, B = 0.01, -0.005


def alternating(n: int = 504) -> pd.Series:
    """Rendimientos que alternan +1% y −0.5%: media, desviación y desviación a la baja conocidas en forma cerrada."""
    return pd.Series([A, B] * (n // 2))


def test_metrics_on_synthetic_curves_with_known_result():
    """Sharpe, Sortino, drawdown, CAGR, Calmar, win rate, profit factor, Sharpe móvil y DSR con resultado conocido."""
    r = alternating()
    n = len(r)
    mean = (A + B) / 2
    std = math.sqrt(n / (n - 1)) * abs(A - B) / 2
    assert sharpe_ratio(r) == pytest.approx(mean / std * math.sqrt(252), rel=1e-12)
    assert sortino_ratio(r) == pytest.approx(mean / (abs(B) / math.sqrt(2)) * math.sqrt(252), rel=1e-12)
    assert math.isnan(sharpe_ratio(pd.Series([0.001] * 10)))
    assert math.isnan(sortino_ratio(pd.Series([0.001] * 10)))

    assert max_drawdown(pd.Series([100.0, 120.0, 90.0, 130.0]), 100.0) == pytest.approx(-0.25)
    assert max_drawdown(pd.Series([95.0, 97.0, 99.0]), 100.0) == pytest.approx(-0.05)
    doubling = pd.Series(100.0 * 2 ** (np.arange(1, 253) / 252))
    assert cagr(doubling, 100.0) == pytest.approx(1.0, rel=1e-12)
    dipped = doubling.copy()
    dipped.iloc[100] = dipped.iloc[99] * 0.9
    assert calmar_ratio(dipped, 100.0) == pytest.approx(1.0 / 0.1, rel=1e-12)

    pnl = pd.Series([100.0, -50.0, 200.0, -25.0])
    assert win_rate(pnl) == 0.5
    assert profit_factor(pnl) == pytest.approx(300 / 75)
    assert profit_factor(pd.Series([10.0, 5.0])) == float("inf")
    assert math.isnan(win_rate(pd.Series([], dtype=float)))

    long_r = alternating(800)
    rolling = rolling_sharpe(long_r, 252).dropna()
    assert len(rolling) == 800 - 251
    np.testing.assert_allclose(rolling.iloc[::2], sharpe_ratio(long_r.iloc[:252]), rtol=1e-9)

    rng = np.random.default_rng(42)
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(rng.standard_normal((2000, 1000)).max(axis=1).mean(),
                                                           abs=0.05)
    assert expected_max_sharpe(1, 1.0) == 0.0
    noisy = pd.Series(np.random.default_rng(1).normal(0.0005, 0.01, 1000))
    one = deflated_sharpe(noisy, 1, 0.0001)
    sr = noisy.mean() / noisy.std(ddof=1)
    assert one["sr0_daily"] == 0.0
    assert one["dsr"] == pytest.approx(probabilistic_sharpe(sr, 0.0, 1000, one["skew"], one["kurtosis"]))
    assert deflated_sharpe(noisy, 1000, 0.0001)["dsr"] < one["dsr"]
