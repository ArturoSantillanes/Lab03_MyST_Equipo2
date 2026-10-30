"""Etapa 2 del embudo: filtro de correlación móvil sobre RETORNOS (heurística, no criterio definitivo).

Dueño: Milca
Lee: log-precios de train.
Escribe: (en memoria) correlación media por sub-ventana de cada subgrupo y si pasa.
Reglas: SOLO train. La correlación es una HEURÍSTICA para descartar grupos que no se mueven juntos;
el criterio definitivo de cointegración es Johansen (etapa 4).

Correlación sobre log-retornos diarios (no sobre precios: dos precios con tendencia salen correlacionados
aunque no tengan relación). Ventana móvil de ROLLING_WINDOW = 63 días hábiles (un trimestre): suficientes
datos para estimar una correlación estable y lo bastante corta para detectar cambios dentro de un año.
Estabilidad: train se divide en N_SUBWINDOWS = 3 sub-ventanas consecutivas de ~3 años (el mismo largo que
test y valid); el subgrupo pasa si en TODAS la correlación media por pares (promedio en el tiempo de la
correlación móvil) supera MIN_MEAN_CORRELATION = 0.7.
Dato SOLO informativo (no sustituye el criterio, que el profesor pidió sobre retornos): la correlación media por
pares de los log-precios en niveles, que sale mucho más alta porque dos series con tendencia se correlacionan.
"""
import itertools

import numpy as np
import pandas as pd

ROLLING_WINDOW = 63
N_SUBWINDOWS = 3
MIN_MEAN_CORRELATION = 0.70


def rolling_pair_correlations(log_prices: pd.DataFrame, window: int = ROLLING_WINDOW) -> dict:
    """Correlación móvil de log-retornos para cada par de activos."""
    returns = log_prices.diff().iloc[1:]
    return {(a, b): returns[a].rolling(window).corr(returns[b]) for a, b in itertools.combinations(returns.columns, 2)}


def subwindow_mean_correlations(group: tuple, pair_corr: dict, n_subwindows: int = N_SUBWINDOWS) -> list[float]:
    """Correlación media por pares del subgrupo en cada sub-ventana consecutiva de train."""
    mean_corr = pd.concat([pair_corr[pair] for pair in itertools.combinations(group, 2)], axis=1).mean(axis=1)
    chunks = np.array_split(np.arange(len(mean_corr)), n_subwindows)
    return [float(mean_corr.iloc[idx].mean()) for idx in chunks]


def mean_level_correlation(group: tuple, log_prices: pd.DataFrame) -> float:
    """INFORMATIVO: correlación media por pares de los log-precios en niveles (no es el criterio de la etapa 2)."""
    corr = log_prices[list(group)].corr().to_numpy()
    return float(corr[np.triu_indices(len(group), k=1)].mean())


def passes_correlation(subwindow_corr: list[float], threshold: float = MIN_MEAN_CORRELATION) -> bool:
    """True si la correlación media supera el umbral en todas las sub-ventanas."""
    return min(subwindow_corr) > threshold
