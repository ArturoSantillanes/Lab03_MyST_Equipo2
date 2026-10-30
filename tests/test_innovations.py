"""Prueba obligatoria 4 (Arturo): las innovaciones normalizadas tienen varianza cercana a 1 y autocorrelación no
significativa.

Criterio declarado (train, después del burn-in de ambos filtros):
    varianza de ỹ_t/√S_t dentro de [0.8, 1.2]
    Ljung-Box con 10 rezagos: p > 0.05 (autocorrelación conjunta no significativa)
    autocorrelación de orden 1 dentro de la banda ±1.96/√n
Se aplica al FILTRO 2, que modela el spread como proceso reversivo: sus innovaciones son errores de pronóstico a un
día y deben ser ruido blanco. Las innovaciones del FILTRO 1 son el propio spread (S_t = ỹ_t), que revierte despacio:
su autocorrelación alta es la señal que se opera, no un error. Se documenta con una prueba aparte.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statsmodels.stats.diagnostic import acorr_ljungbox

from kalman.filter2_spread import run_filter, variance_ratio

KALMAN_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "kalman"
VARIANCE_RANGE = (0.8, 1.2)
LJUNG_BOX_LAGS = 10
SIGNIFICANCE = 0.05


def train_after_burn_in(name: str, column: str) -> np.ndarray:
    """Columna de un artefacto de kalman en train, sin burn-in."""
    if not (KALMAN_DIR / name).exists():
        pytest.skip("correr python main.py primero")
    frame = pd.read_csv(KALMAN_DIR / name, index_col="date", parse_dates=True)
    return frame.loc[(frame["period"] == "train") & ~frame["burn_in"], column].to_numpy()


def assert_white(normalized: np.ndarray) -> None:
    """Varianza cercana a 1, Ljung-Box no significativo y acf1 dentro de la banda."""
    acf1 = np.corrcoef(normalized[1:], normalized[:-1])[0, 1]
    pvalue = acorr_ljungbox(normalized, lags=[LJUNG_BOX_LAGS]).lb_pvalue.iloc[0]
    assert VARIANCE_RANGE[0] <= normalized.var() <= VARIANCE_RANGE[1], f"varianza = {normalized.var():.3f}"
    assert pvalue > SIGNIFICANCE, f"Ljung-Box p = {pvalue:.4f}"
    assert abs(acf1) < 1.96 / np.sqrt(len(normalized)), f"acf1 = {acf1:.4f}"


def test_filter2_normalized_innovations_are_white_in_train():
    assert_white(train_after_burn_in("z_t.csv", "innovation_normalized"))


def test_filter2_innovations_are_white_on_a_simulated_spread_with_known_parameters():
    rng = np.random.default_rng(42)
    phi, sigma_eps, n = 0.95, 0.004, 3000
    mu = np.cumsum(rng.normal(0, 2e-4, n))
    deviation = np.zeros(n)
    for t in range(1, n):
        deviation[t] = phi * deviation[t - 1] + rng.normal(0, sigma_eps)
    kappa = variance_ratio(np.array([phi]))
    params = {"phi": [phi], "kappa": kappa, "Q_mu": 4e-8, "Q_sig2": 0.0, "R_sig2": 2 * sigma_eps ** 4,
              "mu0": 0.0, "pmu0": 1e-4, "sig2_0": kappa * sigma_eps ** 2, "psig0": 1e-12}
    result = run_filter(mu + deviation, params)
    normalized = (result["innovation"] / np.sqrt(result["innovation_var"]))[500:]
    assert_white(normalized)


def test_filter1_innovations_keep_the_spread_reversion():
    # Diagnóstico del filtro 1: sus innovaciones son el spread, que revierte con vida media de semanas, así que
    # tienen autocorrelación alta (≈ 0.97). Si fueran blancas, el filtro 1 ya se habría comido la reversión.
    normalized = train_after_burn_in("spread_t.csv", "innovation_normalized")
    assert np.corrcoef(normalized[1:], normalized[:-1])[0, 1] > 0.5
