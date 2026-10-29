"""Filtro de Kalman 1: pesos dinámicos β̂_t del grupo final y spread S_t, sobre log-precios.

Dueño: Milca
Lee: data/processed/prices.csv, data/processed/splits.json, artifacts/cointegration/selected_group.json
Escribe: artifacts/kalman/beta_t.csv, spread_t.csv, gain_1.csv, posvar_1.csv, filter1_q_grid.csv, filter1_params.json,
         artifacts/figures/fig3_beta_t.png, fig_diagnostico_filtro1.png
Reglas: q, R, P_0 y el burn-in se eligen SOLO con train. Sin aleatoriedad (semilla 42 en main.py).

Notación del profesor:
    estado        z_t = β_t  (pesos de las patas que no son la de normalización; SIN intercepto α)
    transición    z_t = A·z_{t-1} + W_t,   W_t ~ N(0, Q),   A = I (explícita)
    observación   y_t = C_t·z_t + V_t,     V_t ~ N(0, R)
                  y_t = log-precio centrado de la pata de normalización; C_t = log-precios centrados de las demás
    Q = covarianza del ruido de proceso = velocidad del drift (qué tan rápido dejamos cambiar los pesos)
    R = varianza del ruido de observación
    Predecir:     ẑ_{t|t-1} = A·ẑ_{t-1|t-1};   P_{t|t-1} = A·P_{t-1|t-1}·Aᵀ + Q
    Actualizar:   ỹ_t = y_t − C_t·ẑ_{t|t-1};   S_t = C_t·P_{t|t-1}·C_tᵀ + R;   K_t = P_{t|t-1}·C_tᵀ·S_t⁻¹
                  ẑ_{t|t} = ẑ_{t|t-1} + K_t·ỹ_t;   P_{t|t} = (I − K_t·C_t)·P_{t|t-1}
"Centrado" = menos la media de train de cada log-precio: es el término determinístico (constante sin restringir,
igual que Johansen con det_order=0); por eso no hay α en el estado. Aquí S_t es la varianza de la innovación; el
spread se llama S_t solo en el contexto de Johansen (S_t = βᵀP_t).

Pesos normalizados: +1 en la pata de normalización (la de mayor |β| de Johansen) y −ẑ_{t|t-1} en las demás.
Spread del día t con los pesos previos (sin look-ahead): S_t = β̂_tᵀ(P_t − μ_train) = ỹ_t.
El filtro se CORRE hacia adelante sobre toda la muestra con parámetros congelados en train (Arturo y Paula
necesitan las series de test y valid); nada se estima, calibra ni elige con test o valid.

Calibración de Q (NO por máxima verosimilitud libre: en este modelo elige Q demasiado grande, los pesos persiguen
cada desviación y el filtro se come la reversión que se quiere operar):
    Q = q · Q_base, con Q_base = R · diag(1 / E[C_i²] en train)  (cada componente aporta lo mismo a la predicción)
    q en Q_GRID: 29 valores log-espaciados de 1e-8 a 1e-1 (4 por década), de "pesos casi congelados"
    (Q/R ~ 1e-7) a "pesos que persiguen cada desviación" (Q/R ~ 1).
    Criterio fijado ANTES de ver resultados (train, después del burn-in de cada q):
      (a) spread estacionario: ADF p < 0.05
      (b) vida media del spread entre 5 y 60 días
      (c) CERO cambios de signo en los pesos
      (d) los pesos siguen adaptándose: al final de train difieren en al menos MIN_ADAPTATION de los que daría el
          filtro con Q = 0 (pesos congelados)
    Si varios q cumplen: el centro de la meseta (la racha contigua más larga de q que cumplen; su valor central).
    Si ninguno cumple, el grupo no tiene q válido (no hay respaldo).
R = varianza residual de la regresión OLS estática (sin constante, log-precios centrados) en train.
P_0 = P0_DIFFUSE · I: con el vector normalizado todos los |β_i| <= 1, así que una desviación de 100 es difusa.
Burn-in: primer día desde el cual la varianza posterior de TODOS los componentes se queda por debajo del techo de su
banda estable hasta el final de train; banda = exp(mediana ± 3·MAD) del log de P_t|t en train.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from cointegration.figures import plot_beta, plot_filter_diagnostic, ticker_colors
from cointegration.spread_checks import HALF_LIFE_RANGE, MAX_ADF_PVALUE, adf_test, half_life
from data.split import assign_period, slice_period

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.json"
SELECTION_PATH = ROOT / "artifacts" / "cointegration" / "selected_group.json"
OUT_DIR = ROOT / "artifacts" / "kalman"
Q_GRID = np.logspace(-8, -1, 29)
MIN_ADAPTATION = 0.05
P0_DIFFUSE = 1e4
P0_ROBUSTNESS = (1e2, 1e4, 1e6)
BAND_MAD_K = 3.0
MAD_TO_SD = 1.4826
LJUNG_BOX_LAGS = 10


def observation_design(centered: pd.DataFrame, leg: str) -> tuple[np.ndarray, np.ndarray]:
    """y_t (pata de normalización) y C_t (las demás patas), ya centrados."""
    return centered[leg].to_numpy(), centered.drop(columns=leg).to_numpy()


def kalman_filter(y: np.ndarray, C: np.ndarray, Q: np.ndarray, R: float, z0: np.ndarray, P0: np.ndarray) -> dict:
    """Ecuaciones del profesor con A = I; devuelve ẑ_{t|t-1}, ỹ_t, S_t, K_t, ẑ_{t|t} y diag(P_{t|t}) por fecha."""
    n, k = C.shape
    A, identity = np.eye(k), np.eye(k)
    z, P = z0.astype(float).copy(), P0.astype(float).copy()
    out = {name: np.empty((n, k)) for name in ("z_prior", "z_post", "gain", "posvar")}
    out["innovation"], out["innovation_var"] = np.empty(n), np.empty(n)
    for t in range(n):
        z_prior = A @ z
        P_prior = A @ P @ A.T + Q
        c = C[t]
        innovation = y[t] - c @ z_prior
        S = c @ P_prior @ c + R
        K = P_prior @ c / S
        z = z_prior + K * innovation
        P = (identity - np.outer(K, c)) @ P_prior
        out["z_prior"][t], out["innovation"][t], out["innovation_var"][t] = z_prior, innovation, S
        out["gain"][t], out["z_post"][t], out["posvar"][t] = K, z, np.diag(P)
    return out


def observation_noise(y: np.ndarray, C: np.ndarray) -> float:
    """R = varianza residual de la regresión OLS estática de y sobre C (sin constante) en train."""
    coef = np.linalg.lstsq(C, y, rcond=None)[0]
    residual = y - C @ coef
    return float(residual @ residual / (len(y) - C.shape[1]))


def state_noise_base(C_train: np.ndarray, R: float) -> np.ndarray:
    """Q_base = R · diag(1 / E[C_i²]): cada componente aporta R a la varianza de la predicción por unidad de q."""
    return R * np.diag(1.0 / np.mean(C_train ** 2, axis=0))


def stable_band(posvar_train: np.ndarray) -> np.ndarray:
    """Banda estable por componente: exp(mediana ± BAND_MAD_K·MAD) del log de P (filas: inferior, superior)."""
    log_p = np.log(posvar_train)
    median = np.median(log_p, axis=0)
    mad = MAD_TO_SD * np.median(np.abs(log_p - median), axis=0)
    return np.exp(np.vstack([median - BAND_MAD_K * mad, median + BAND_MAD_K * mad]))


def measure_burn_in(posvar_train: np.ndarray) -> int:
    """Primer día desde el cual TODOS los componentes se quedan bajo el techo de su banda hasta el final de train."""
    inside = (posvar_train <= stable_band(posvar_train)[1]).all(axis=1)
    stays = np.flip(np.cumprod(np.flip(inside))).astype(bool)
    return int(np.argmax(stays))


def weights_from_state(z_prior: np.ndarray, tickers: list[str], leg: str) -> pd.DataFrame:
    """β̂_t normalizado por activo: +1 en la pata de normalización y −ẑ en las demás."""
    weights = pd.DataFrame(-z_prior, columns=[t for t in tickers if t != leg])
    weights[leg] = 1.0
    return weights[tickers]


def sign_changes(weights: np.ndarray) -> np.ndarray:
    """Cambios de signo de cada peso entre reestimaciones consecutivas."""
    return (np.diff(np.sign(weights), axis=0) != 0).sum(axis=0)


def innovation_diagnostics(innovation: np.ndarray, innovation_var: np.ndarray) -> dict:
    """Varianza y autocorrelación de orden 1 de las innovaciones normalizadas ỹ/√S, y Ljung-Box."""
    normalized = innovation / np.sqrt(innovation_var)
    return {"innovation_norm_var": float(normalized.var()),
            "innovation_norm_acf1": float(np.corrcoef(normalized[1:], normalized[:-1])[0, 1]),
            "ljung_box_pvalue": float(acorr_ljungbox(normalized, lags=[LJUNG_BOX_LAGS]).lb_pvalue.iloc[0])}


def evaluate_q(y: np.ndarray, C: np.ndarray, Q: np.ndarray, R: float, z0: np.ndarray, frozen_end: np.ndarray) -> dict:
    """Métricas del criterio para un Q (train, después del burn-in de ese Q)."""
    result = kalman_filter(y, C, Q, R, z0, P0_DIFFUSE * np.eye(len(z0)))
    burn_in = measure_burn_in(result["posvar"])
    spread = pd.Series(result["innovation"][burn_in:])
    stat, pvalue = adf_test(spread)
    return {"burn_in_days": burn_in, "spread_adf_stat": stat, "spread_adf_pvalue": pvalue,
            "spread_half_life_days": half_life(spread),
            "sign_changes": int(sign_changes(-result["z_prior"][burn_in:]).sum()),
            "adaptation_vs_frozen": float(np.abs(result["z_prior"][-1] - frozen_end).max()),
            **innovation_diagnostics(result["innovation"][burn_in:], result["innovation_var"][burn_in:])}


def q_grid(y: np.ndarray, C: np.ndarray, R: float, z0: np.ndarray) -> pd.DataFrame:
    """Barre Q_GRID y marca qué q cumplen cada parte del criterio."""
    Q_base = state_noise_base(C, R)
    frozen_end = kalman_filter(y, C, np.zeros_like(Q_base), R, z0, P0_DIFFUSE * np.eye(len(z0)))["z_prior"][-1]
    table = pd.DataFrame([{"q": q, "Q_over_R": float(np.diag(q * Q_base).mean() / R),
                           **evaluate_q(y, C, q * Q_base, R, z0, frozen_end)} for q in Q_GRID])
    return table.assign(
        ok_a_stationary=table["spread_adf_pvalue"] < MAX_ADF_PVALUE,
        ok_b_half_life=table["spread_half_life_days"].between(*HALF_LIFE_RANGE),
        ok_c_no_sign_change=table["sign_changes"] == 0,
        ok_d_adapts=table["adaptation_vs_frozen"] >= MIN_ADAPTATION,
    ).assign(ok_all=lambda t: t.filter(like="ok_").all(axis=1))


def plateau_center(passing: pd.Series) -> int | None:
    """Posición central de la racha contigua más larga de valores True; None si no hay ninguno."""
    runs, current = [], []
    for i, ok in enumerate(passing):
        current = current + [i] if ok else []
        if current:
            runs.append(list(current))
    if not runs:
        return None
    longest = max(runs, key=len)
    return longest[len(longest) // 2]


def calibrate_group(log_prices_train: pd.DataFrame, beta: dict, leg: str) -> dict:
    """Calibra R, la rejilla de q y el burn-in de un grupo con SOLO train; q = None si ningún q cumple (a)-(d)."""
    tickers = list(log_prices_train.columns)
    others = [t for t in tickers if t != leg]
    train_mean = log_prices_train.mean()
    y, C = observation_design(log_prices_train - train_mean, leg)
    z0 = -np.array([beta[t] for t in others])
    R = observation_noise(y, C)
    grid = q_grid(y, C, R, z0)
    position = plateau_center(grid["ok_all"])
    calibration = {"tickers": tickers, "leg": leg, "others": others, "train_mean": train_mean, "z0": z0, "R": R,
                   "grid": grid, "q": None}
    if position is None:
        return calibration
    q = float(grid.iloc[position]["q"])
    Q = q * state_noise_base(C, R)
    result = kalman_filter(y, C, Q, R, z0, P0_DIFFUSE * np.eye(len(z0)))
    burn_in = measure_burn_in(result["posvar"])
    calibration.update(q=q, Q=Q, burn_in=burn_in, metrics=grid.iloc[position].to_dict(),
                       sign_changes_by_asset=dict(zip(others, sign_changes(-result["z_prior"][burn_in:]).tolist())),
                       posvar_floor=dict(zip(others, np.median(result["posvar"][burn_in:], axis=0).tolist())),
                       posvar_train=result["posvar"], y=y, C=C)
    return calibration


def p0_robustness(y: np.ndarray, C: np.ndarray, Q: np.ndarray, R: float, z0: np.ndarray, burn_in: int) -> dict:
    """Corre el filtro con otros P_0 y mide la mayor diferencia de pesos y spread después del burn-in."""
    runs = {v: kalman_filter(y, C, Q, R, z0, v * np.eye(len(z0))) for v in P0_ROBUSTNESS}
    base = runs[P0_DIFFUSE]
    return {f"{v:g}": {"max_abs_weight_diff": float(np.abs(r["z_prior"][burn_in:] - base["z_prior"][burn_in:]).max()),
                       "max_abs_spread_diff": float(np.abs(r["innovation"][burn_in:] - base["innovation"][burn_in:]).max()),
                       "burn_in_days": measure_burn_in(r["posvar"])} for v, r in runs.items()}


def build_outputs(result: dict, centered: pd.DataFrame, leg: str, period: pd.Series, burn_in: int) -> dict:
    """DataFrames de salida con fecha, columna period y marca de burn-in."""
    tickers = list(centered.columns)
    others = [t for t in tickers if t != leg]
    innovation_std = np.sqrt(result["innovation_var"])
    frames = {
        "beta_t": weights_from_state(result["z_prior"], tickers, leg),
        "spread_t": pd.DataFrame({"spread": result["innovation"], "innovation_normalized": result["innovation"] / innovation_std,
                                  "innovation_std": innovation_std}),
        "gain_1": pd.DataFrame(result["gain"], columns=others),
        "posvar_1": pd.DataFrame(result["posvar"], columns=others),
    }
    for frame in frames.values():
        frame.index = centered.index
        frame["period"] = period.to_numpy()
        frame["burn_in"] = np.arange(len(frame)) < burn_in
    return frames


def run() -> None:
    """Calibra en train, corre el filtro hacia adelante en toda la muestra con parámetros congelados y guarda todo."""
    log_prices = np.log(pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True).drop(columns="period"))
    splits = json.loads(SPLITS_PATH.read_text())
    selection = json.loads(SELECTION_PATH.read_text())
    tickers, leg = selection["tickers"], selection["normalization_leg"]
    calibration = calibrate_group(slice_period(log_prices[tickers], splits, "train"), selection["beta"], leg)
    if calibration["q"] is None:
        raise RuntimeError("Ningún q cumple (a)-(d) para el grupo final; ver filter1_q_grid.csv (no se relajan criterios)")
    others, R, Q, z0, burn_in = calibration["others"], calibration["R"], calibration["Q"], calibration["z0"], calibration["burn_in"]

    centered = log_prices[tickers] - calibration["train_mean"]
    y, C = observation_design(centered, leg)
    result = kalman_filter(y, C, Q, R, z0, P0_DIFFUSE * np.eye(len(z0)))
    n_train = len(calibration["y"])
    frames = build_outputs(result, centered, leg, assign_period(centered.index, splits), burn_in)
    grid = calibration["grid"]
    params = {
        "period_used": "train",
        "state": "z_t = beta_t (pesos sin intercepto); A = I",
        "tickers": tickers, "normalization_leg": leg,
        "train_mean_log_price": calibration["train_mean"].to_dict(),
        "R": R, "R_daily_std_log": float(np.sqrt(R)),
        "q_grid_size": len(grid), "q_passing": int(grid["ok_all"].sum()), "q": calibration["q"],
        "q_on_grid_edge": bool(calibration["q"] in (Q_GRID.min(), Q_GRID.max())), "fallback_count": 0,
        "Q_over_R": float(calibration["metrics"]["Q_over_R"]), "Q_diag": dict(zip(others, np.diag(Q).tolist())),
        "P0": f"{P0_DIFFUSE:g} · I", "z0": dict(zip(others, z0.tolist())),
        "burn_in_days": burn_in, "burn_in_share_of_train": burn_in / n_train,
        "burn_in_end": centered.index[burn_in].strftime("%Y-%m-%d"),
        "sign_changes_by_asset_train": calibration["sign_changes_by_asset"],
        "posvar_floor_median_after_burn_in": calibration["posvar_floor"],
        "posvar_min_train": float(calibration["posvar_train"].min()),
        "chosen_q_metrics": {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v))
                             for k, v in calibration["metrics"].items()},
        "p0_robustness": p0_robustness(calibration["y"], calibration["C"], Q, R, z0, burn_in),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", date_format="%Y-%m-%d")
    grid.to_csv(OUT_DIR / "filter1_q_grid.csv", index=False)
    (OUT_DIR / "filter1_params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False))
    plot_beta(frames["beta_t"], selection["beta"], splits)
    train_rows = frames["posvar_1"]["period"] == "train"
    plot_filter_diagnostic(frames["posvar_1"].loc[train_rows, others], frames["gain_1"].loc[train_rows, others],
                           centered.index[burn_in], stable_band(calibration["posvar_train"]), ticker_colors(tickers))

    m = params["chosen_q_metrics"]
    print(f"  q = {params['q']:.2e} ({params['q_passing']} de {len(grid)} cumplen (a)-(d); orilla: {params['q_on_grid_edge']}); "
          f"R = {R:.3e}; burn-in = {burn_in} días ({params['burn_in_share_of_train']:.0%} de train)")
    print(f"  spread (train, tras burn-in): ADF p = {m['spread_adf_pvalue']:.2e}, vida media = "
          f"{m['spread_half_life_days']:.1f} días, cambios de signo = {sum(params['sign_changes_by_asset_train'].values())}")
