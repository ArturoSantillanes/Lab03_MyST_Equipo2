"""VECM del grupo elegido con el vector cointegrante de Johansen como término de corrección de error.

Dueño: Arturo
Lee: data/processed/prices.csv (log-precios de los 6 activos), artifacts/cointegration/selected_group.json (β de
     Johansen, train), artifacts/kalman/beta_t.csv y filter1_params.json (β̂_t y media de train del filtro 1)
Escribe: (lo guarda strategy.run) artifacts/strategy/vecm_summary.json
Reglas: se estima SOLO con train. Log-precios (los pesos son exposición en valor).

Del VAR(p) al VECM:
    VAR(p):  P_t = c + A_1·P_{t−1} + ... + A_p·P_{t−p} + ε_t
    Restando P_{t−1} y reacomodando:
             ΔP_t = c + Π·P_{t−1} + Σ_{i=1}^{p−1} Γ_i·ΔP_{t−i} + ε_t,   Π = Σ_i A_i − I,   Γ_i = −Σ_{j>i} A_j
    Con r relaciones de cointegración, Π = α·βᵀ (rango r): βᵀP_{t−1} es el desequilibrio de ayer (término de
    corrección de error, ECT) y α dice cuánto corrige cada activo por día.
Especificación (la misma que usó Milca en Johansen): p = 1, es decir k_ar_diff = p − 1 = 0 rezagos de ΔP, r = 1 y
constante sin restringir (fuera de la relación, caso 3). Se verifica con select_order de statsmodels en train.
β se FIJA en el vector de Johansen de train (no se reestima), y α, c y Γ se estiman por MCO ecuación por ecuación
(statsmodels OLS): dado β, ese es el estimador de máxima verosimilitud del VECM. Como control se reporta también el β
que estima libremente la clase VECM de statsmodels.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.vector_ar.vecm import VECM, select_order

MAX_LAGS = 10
DETERMINISTIC = "co"


def lag_order(log_prices_train: pd.DataFrame) -> dict:
    """Número de rezagos de ΔP (k_ar_diff) que eligen AIC, BIC, HQIC y FPE en train."""
    selection = select_order(log_prices_train.to_numpy(), maxlags=MAX_LAGS, deterministic=DETERMINISTIC)
    return {name: int(value) for name, value in selection.selected_orders.items()}


def design_matrix(log_prices: pd.DataFrame, beta: pd.Series, k_ar_diff: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ΔP_t (respuestas) y [constante, ECT_{t−1}, ΔP_{t−1..t−k}] (regresores), alineados por fecha."""
    delta = log_prices.diff()
    regressors = pd.DataFrame({"const": 1.0, "ect_lag": (log_prices @ beta).shift(1)}, index=log_prices.index)
    for lag in range(1, k_ar_diff + 1):
        regressors = regressors.join(delta.shift(lag).add_suffix(f"_dlag{lag}"))
    rows = regressors.notna().all(axis=1) & delta.notna().all(axis=1)
    return delta[rows], regressors[rows]


def fit_vecm(log_prices_train: pd.DataFrame, beta: pd.Series, k_ar_diff: int) -> dict:
    """α, c y Γ por MCO con β fijo; residuales, matriz de covarianza y vida media implícita del desequilibrio."""
    delta, regressors = design_matrix(log_prices_train, beta, k_ar_diff)
    fits = {ticker: sm.OLS(delta[ticker], regressors).fit() for ticker in delta.columns}
    alpha = pd.Series({t: f.params["ect_lag"] for t, f in fits.items()})
    residuals = pd.DataFrame({t: f.resid for t, f in fits.items()})
    speed = float(beta @ alpha)
    return {
        "alpha": alpha, "alpha_tvalues": pd.Series({t: f.tvalues["ect_lag"] for t, f in fits.items()}),
        "const": pd.Series({t: f.params["const"] for t, f in fits.items()}),
        "gamma": {t: f.params.drop(["const", "ect_lag"]).to_dict() for t, f in fits.items()},
        "r_squared": pd.Series({t: f.rsquared for t, f in fits.items()}),
        "residuals": residuals, "sigma_u": residuals.cov(),
        "ect_speed": speed, "ect_half_life_days": float(-np.log(2) / np.log(1 + speed)) if -1 < speed < 0 else float("inf"),
        "n_obs": len(delta),
    }


def free_johansen_beta(log_prices_train: pd.DataFrame, k_ar_diff: int, leg: str) -> pd.Series:
    """β que estima la clase VECM de statsmodels sin fijarlo, normalizado en la pata de normalización (control)."""
    result = VECM(log_prices_train.to_numpy(), k_ar_diff=k_ar_diff, coint_rank=1, deterministic=DETERMINISTIC).fit()
    beta = pd.Series(result.beta[:, 0], index=log_prices_train.columns)
    return beta / beta[leg]


def equilibrium_error(log_prices: pd.DataFrame, beta_t: pd.DataFrame, train_mean: pd.Series) -> pd.Series:
    """Residual de equilibrio del VECM con los pesos dinámicos: ECT_t = β̂_tᵀ(P_t − μ_train).

    Es el término de corrección de error del VECM evaluado con el β̂_t del filtro 1 (pesos previos, sin look-ahead);
    la media de train es la constante sin restringir. Coincide con la columna spread de spread_t.csv.
    """
    centered = log_prices[beta_t.columns] - train_mean[beta_t.columns]
    return (centered * beta_t).sum(axis=1).rename("ect")
