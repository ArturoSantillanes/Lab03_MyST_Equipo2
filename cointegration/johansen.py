"""Etapa 4 del embudo: prueba de Johansen (traza y máximo eigenvalor), orden de rezagos y rango.

Dueño: Milca
Lee: log-precios de train del subgrupo.
Escribe: (en memoria) estadísticos, valores críticos, eigenvalores, eigenvectores y r̂.
Reglas: SOLO train.

Especificación determinística: DET_ORDER = 0 en statsmodels = constante SIN RESTRINGIR (caso 3 de Johansen):
se resta la media a los niveles y a las diferencias, así que la constante queda FUERA de la relación de
cointegración, en los términos determinísticos del VECM. Es consistente con el filtro de Kalman 1, que no
lleva intercepto en el estado y trata la constante igual (resta la media de train de cada log-precio).
Valores críticos: MacKinnon, Haug y Michelis (1996), tablas de statsmodels.tsa.coint_tables.

Orden p del VAR en niveles: se calcula con AIC y con BIC (hasta MAX_LAGS); se USA el de BIC, regla fijada
antes de ver resultados: con 6 series cada rezago agrega 36 parámetros, y rezagos de más le quitan potencia
a la prueba de Johansen. El VECM usa p − 1 rezagos en diferencias.
r̂ = número de rechazos consecutivos de "r <= i" con la traza al 95%; se reporta también el del máximo eigenvalor.
"""
import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from statsmodels.tsa.vector_ar.vecm import coint_johansen

DET_ORDER = 0
DETERMINISTIC = "constante sin restringir (det_order=0, caso 3): fuera de la relación de cointegración"
CRITICAL_VALUES_SOURCE = "MacKinnon, Haug y Michelis (1996), statsmodels.tsa.coint_tables"
MAX_LAGS = 10
CV_95, CV_99 = 1, 2


def var_lag_orders(log_prices: pd.DataFrame, max_lags: int = MAX_LAGS) -> dict:
    """Orden p del VAR en niveles según AIC y BIC (mínimo 1)."""
    order = VAR(log_prices.to_numpy()).select_order(max_lags)
    return {"aic": max(int(order.aic), 1), "bic": max(int(order.bic), 1)}


def sequential_rank(stats: np.ndarray, critical: np.ndarray) -> int:
    """Rechazos consecutivos de 'r <= i' desde i = 0."""
    return int(np.cumprod(stats > critical).sum())


def johansen_test(log_prices: pd.DataFrame, var_order: int) -> dict:
    """Johansen con DET_ORDER y p − 1 rezagos en diferencias; devuelve estadísticos, valores críticos y vectores."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", np.exceptions.ComplexWarning)  # eigenvalores con parte imaginaria 0
        result = coint_johansen(log_prices.to_numpy(), DET_ORDER, var_order - 1)
    return {
        "var_order": var_order,
        "eigenvalues": np.real(result.eig).tolist(),
        "trace_stats": result.lr1.tolist(),
        "trace_crit_95": result.cvt[:, CV_95].tolist(),
        "trace_crit_99": result.cvt[:, CV_99].tolist(),
        "maxeig_stats": result.lr2.tolist(),
        "maxeig_crit_95": result.cvm[:, CV_95].tolist(),
        "maxeig_crit_99": result.cvm[:, CV_99].tolist(),
        "rank_trace": sequential_rank(result.lr1, result.cvt[:, CV_95]),
        "rank_maxeig": sequential_rank(result.lr2, result.cvm[:, CV_95]),
        "eigenvectors": np.real(result.evec),
    }


def rank_alarm(rank: int, n_assets: int) -> str:
    """Mensaje de alarma si r̂ = 0 (no hay relación) o r̂ = n (las series ya serían estacionarias)."""
    if rank == 0:
        return "ALARMA r̂ = 0: no hay ninguna combinación estacionaria; el grupo no está cointegrado"
    if rank == n_assets:
        return "ALARMA r̂ = n: Π tiene rango completo, cada serie sería estacionaria por sí sola; no es cointegración"
    return ""
