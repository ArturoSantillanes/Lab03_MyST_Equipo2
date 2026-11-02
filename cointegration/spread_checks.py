"""Etapas 5 y 6 del embudo: selección del vector, normalización y validación del spread.

Dueño: Milca
Lee: log-precios de train y eigenvectores de Johansen.
Escribe: (en memoria) tabla de columnas con ADF y vida media, vector elegido y normalizado.
Reglas: SOLO train.

Spread de cada columna: S_t = βᵀ·P_t con P_t = log-precios. Vida media: ajuste ΔS_t = a + b·S_{t-1} + e;
φ = 1 + b y vida media = −ln 2 / ln φ (infinita si no hay reversión, φ ≥ 1).
Etapa 5: se elige, entre las r̂ columnas cointegradas, la de ADF más negativo con vida media operable.
Etapa 6: el spread elegido debe tener ADF p < MAX_ADF_PVALUE y vida media en HALF_LIFE_RANGE días.
Normalización: se divide el vector entre su coeficiente de mayor valor absoluto; esa pata queda en +1.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

MAX_ADF_PVALUE = 0.05
HALF_LIFE_RANGE = (5.0, 60.0)


def spread_series(log_prices: pd.DataFrame, beta: np.ndarray) -> pd.Series:
    """S_t = βᵀ·P_t."""
    return pd.Series(log_prices.to_numpy() @ beta, index=log_prices.index, name="spread")


def adf_test(spread: pd.Series) -> tuple[float, float]:
    """Estadístico y p-value del ADF (con constante, rezagos por AIC)."""
    stat, pvalue, *_ = adfuller(spread.to_numpy(), regression="c", autolag="AIC", result_object=False)
    return float(stat), float(pvalue)


def half_life(spread: pd.Series) -> float:
    """Vida media en días; inf si el spread no revierte a la media."""
    lagged, delta = spread.shift(1).iloc[1:].to_numpy(), spread.diff().iloc[1:].to_numpy()
    _, b = np.linalg.lstsq(np.column_stack([np.ones_like(lagged), lagged]), delta, rcond=None)[0]
    phi = 1.0 + b
    return float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else float("inf")


def is_operable(adf_pvalue: float, half_life_days: float) -> bool:
    """Etapa 6: estacionario y con vida media utilizable."""
    return adf_pvalue < MAX_ADF_PVALUE and HALF_LIFE_RANGE[0] <= half_life_days <= HALF_LIFE_RANGE[1]


def column_table(log_prices: pd.DataFrame, eigenvectors: np.ndarray, rank: int) -> pd.DataFrame:
    """ADF y vida media del spread de CADA columna de β, marcando cuáles están dentro de r̂."""
    rows = []
    for j in range(eigenvectors.shape[1]):
        spread = spread_series(log_prices, eigenvectors[:, j])
        stat, pvalue = adf_test(spread)
        hl = half_life(spread)
        rows.append({"column": j, "within_rank": j < rank, "adf_stat": stat, "adf_pvalue": pvalue,
                     "half_life_days": hl, "operable": is_operable(pvalue, hl)})
    return pd.DataFrame(rows)


def choose_column(table: pd.DataFrame) -> int | None:
    """Columna dentro de r̂ con ADF más negativo y vida media operable; None si ninguna cumple."""
    candidates = table[table["within_rank"] & table["operable"]]
    return None if candidates.empty else int(candidates.loc[candidates["adf_stat"].idxmin(), "column"])


def normalize(beta: np.ndarray) -> tuple[np.ndarray, int]:
    """Divide entre el coeficiente de mayor |β|; devuelve (vector normalizado, índice de esa pata)."""
    leg = int(np.argmax(np.abs(beta)))
    return beta / beta[leg], leg
