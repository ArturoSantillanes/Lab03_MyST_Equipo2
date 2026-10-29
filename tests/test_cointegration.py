"""Prueba 2 de Milca: ADF y vida media dan el resultado correcto en series sintéticas conocidas (semilla 42)."""
import math

import numpy as np
import pandas as pd
import pytest

from cointegration.spread_checks import adf_test, half_life

SEED = 42
PHI = 0.9


def stationary_ar1(n: int = 5000) -> pd.Series:
    """AR(1) estacionario con φ = 0.9: vida media teórica −ln 2 / ln 0.9 ≈ 6.6 días."""
    rng = np.random.default_rng(SEED)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = PHI * x[t - 1] + rng.normal()
    return pd.Series(x)


def random_walk(n: int = 5000) -> pd.Series:
    """Caminata aleatoria (raíz unitaria): no revierte a la media."""
    return pd.Series(np.cumsum(np.random.default_rng(SEED).normal(size=n)))


def test_adf_rejects_unit_root_for_stationary_series():
    _, pvalue = adf_test(stationary_ar1())
    assert pvalue < 0.01


def test_adf_does_not_reject_for_random_walk():
    _, pvalue = adf_test(random_walk())
    assert pvalue > 0.05


def test_half_life_matches_theory_for_stationary_series():
    assert half_life(stationary_ar1()) == pytest.approx(-math.log(2) / math.log(PHI), rel=0.1)


def test_half_life_is_very_long_or_infinite_for_random_walk():
    assert half_life(random_walk()) > 500
