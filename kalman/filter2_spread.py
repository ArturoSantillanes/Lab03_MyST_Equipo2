"""Filtro de Kalman 2: media latente μ_t y varianza latente σ_t² del spread, y z-score en tiempo real.

Dueño: Arturo
Lee: artifacts/kalman/spread_t.csv (columna spread = S_t del filtro 1, con period y burn_in)
Escribe: artifacts/kalman/gain_2.csv, posvar_2.csv, z_t.csv, filter2_params.json
Reglas: φ, R, Q, valores iniciales y burn-in se determinan SOLO con train (después del burn-in del filtro 1). El
z-score usa la media y la varianza PREVIAS a la actualización, nunca las posteriores. Sin aleatoriedad.

Modelo (notación del profesor, mismas ecuaciones que el filtro 1, con A = 1):
    El spread revierte a una media que puede moverse despacio:  S_t − μ_t = Σ_i φ_i·(S_{t−i} − μ_{t−i}) + ε_t
    Estado:        μ_t (media latente)   y   σ_t² (varianza latente del spread alrededor de μ_t)
    Transición:    μ_t = μ_{t−1} + W_t^μ,  W^μ ~ N(0, Q_μ);     σ_t² = σ_{t−1}² + W_t^σ,  W^σ ~ N(0, Q_σ)
    Observación μ: y_t = S_t − Σ_i φ_i·S_{t−i} = C_μ·μ_t + V_t,  C_μ = 1 − Σ_i φ_i,  V_t = ε_t ~ N(0, R_t)
                   con R_t = σ²_{t|t−1}/κ (la varianza del ruido sale de la creencia previa sobre σ_t²)
    Observación σ²: u_t = ỹ_t² − C_μ²·P^μ_{t|t−1} = C_σ·σ_t² + V_t^σ,  C_σ = 1/κ,  V^σ ~ N(0, R_σ)
                   (E[ỹ_t²] = C_μ²·P^μ_{t|t−1} + σ_ε², así que u_t es una medición insesgada de σ_ε² = σ_t²/κ)
    κ = γ_0/σ_ε²: cuántas veces la varianza del spread excede la del choque diario ε_t (proceso AR(p)).
    Predecir:      x̂_{t|t−1} = A·x̂_{t−1|t−1};   P_{t|t−1} = A·P_{t−1|t−1}·Aᵀ + Q
    Actualizar:    ỹ_t = y_t − C·x̂_{t|t−1};   S_t = C·P_{t|t−1}·Cᵀ + R;   K_t = P_{t|t−1}·Cᵀ·S_t⁻¹
                   x̂_{t|t} = x̂_{t|t−1} + K_t·ỹ_t;   P_{t|t} = (1 − K_t·C)·P_{t|t−1}
    Salida:        z_t = (S_t − μ̂_{t|t−1}) / σ̂_{t|t−1}   (estadísticos PREVIOS; ver prior_zscore)

Por qué la observación de μ es y_t y no S_t: el spread está autocorrelacionado (φ cerca de 1). Si se observara S_t
directo, las innovaciones heredarían esa autocorrelación o el filtro tendría que perseguir a S_t (y se comería la
reversión). Con y_t, la innovación ỹ_t es el error de pronóstico a un día del AR(p): debe ser ruido blanco.

Calibración (solo train, después del burn-in del filtro 1):
    φ y p: AR(p) por MCO con constante; p parte del BIC (1..MAX_AR_LAG) y sube al menor orden con residuales de
           train blancos (verificación de Box-Jenkins: Ljung-Box(10) p > 0.05 y |acf1| < 1.96/√n).
    R_σ = varianza muestral de ε_t² (residuales del AR en train): varianza del ruido de la medición de σ_ε².
    Q_μ = q_μ·Var(S_train) y Q_σ = q_σ·(σ_0²)²; (q_μ, q_σ) maximizan la verosimilitud gaussiana de las innovaciones
    en una rejilla que incluye q = 0. Aquí la máxima verosimilitud es adecuada (a diferencia del filtro 1) porque el
    término AR ya explica la reversión: una μ que persiguiera a S_t no mejora el pronóstico a un día.
Inicialización (creencias a priori de train): μ_0 = 0 (los log-precios del filtro 1 están centrados en su media de
train), P^μ_0 = Var(S_train); σ_0² = κ·σ_ε² del AR de train, P^σ_0 = (σ_0²)² (desviación estándar de la creencia =
100% del valor).
Burn-in (train): P^μ_{t|t} dentro de su banda estable (kalman.filter1_weights.measure_burn_in) y P^σ_{t|t} a menos de
1% de su equilibrio de Riccati (P^σ no depende de los datos y converge a un valor fijo); manda el más tardío.
El filtro arranca el primer día después del burn-in del filtro 1 y se CORRE hacia adelante en toda la muestra con
parámetros congelados en train; nada se estima ni elige con test o valid.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima_process import arma_acovf

from kalman.filter1_weights import innovation_diagnostics, measure_burn_in

ROOT = Path(__file__).resolve().parent.parent
SPREAD_PATH = ROOT / "artifacts" / "kalman" / "spread_t.csv"
OUT_DIR = ROOT / "artifacts" / "kalman"
MAX_AR_LAG = 10
WHITE_PVALUE = 0.05
SIGMA_CONVERGENCE_TOL = 0.01
Q_GRID = np.concatenate([[0.0], np.logspace(-8, 0, 33)])
PERIODS = ("train", "test", "valid")


def residuals_are_white(residuals: np.ndarray) -> bool:
    """Chequeo de Box-Jenkins: Ljung-Box(10) p > 0.05 y |acf1| dentro de la banda ±1.96/√n."""
    check = innovation_diagnostics(residuals, np.ones_like(residuals))
    return check["ljung_box_pvalue"] > WHITE_PVALUE and abs(check["innovation_norm_acf1"]) < 1.96 / np.sqrt(len(residuals))


def fit_ar(spread_train: pd.Series) -> dict:
    """AR(p) del spread en train, φ por MCO con constante.

    p: se parte del orden de menor BIC (1..MAX_AR_LAG) y, como en la verificación de Box-Jenkins, se sube al menor
    orden cuyos residuales de train son blancos (residuals_are_white). Devuelve φ, σ_ε², residuales y la tabla de BIC.
    """
    values = spread_train.to_numpy()
    selection = ar_select_order(values, maxlag=MAX_AR_LAG, ic="bic", trend="c")
    bic_order = max(selection.ar_lags) if selection.ar_lags else 1
    for order in range(bic_order, MAX_AR_LAG + 1):
        fit = AutoReg(values, lags=order, trend="c").fit()
        if residuals_are_white(fit.resid):
            break
    else:
        raise RuntimeError(f"Ningún AR(p) con p <= {MAX_AR_LAG} deja residuales blancos en train")
    bic = {int(lags[-1]) if lags else 0: float(value) for lags, value in selection.bic.items()}
    return {"order": order, "bic_order": bic_order, "phi": fit.params[1:], "sigma2_eps": float(fit.sigma2),
            "residuals": fit.resid, "bic_by_order": dict(sorted(bic.items()))}


def variance_ratio(phi: np.ndarray) -> float:
    """κ = γ_0/σ_ε² del AR(p) con coeficientes φ."""
    return float(arma_acovf(np.r_[1.0, -phi], np.array([1.0]), nobs=1, sigma2=1.0)[0])


def ar_observation(spread: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """y_t = S_t − Σ_i φ_i·S_{t−i}; NaN en los primeros p días (faltan rezagos)."""
    y = np.full(len(spread), np.nan)
    for t in range(len(phi), len(spread)):
        y[t] = spread[t] - phi @ spread[t - len(phi):t][::-1]
    return y


def kalman_update(x_prior: float, P_prior: float, y: float, C: float, R: float) -> tuple:
    """Actualización escalar del profesor; devuelve (ỹ_t, S_t, K_t, x̂_{t|t}, P_{t|t})."""
    innovation = y - C * x_prior
    S = C * P_prior * C + R
    K = P_prior * C / S
    return innovation, S, K, x_prior + K * innovation, (1.0 - K * C) * P_prior


def prior_zscore(spread, mu_prev, sigma_prev):
    """z_t = (S_t − μ̂_{t|t−1}) / σ̂_{t|t−1}.

    Se usan la media y la varianza PREVIAS a la actualización del día t (las del cierre de t−1 propagadas un paso).
    Normalizar con la media posterior μ̂_{t|t} usaría la observación S_t para juzgar a la propia S_t: la media
    posterior ya se movió hacia S_t, el z-score sale encogido y eso es look-ahead que invalida el backtest.
    """
    return (spread - mu_prev) / sigma_prev


def run_filter(spread: np.ndarray, params: dict) -> dict:
    """Corre el filtro 2 sobre el spread con parámetros congelados; devuelve las series por día (NaN sin rezagos)."""
    phi, kappa = np.asarray(params["phi"]), params["kappa"]
    C_mu, C_sig = 1.0 - phi.sum(), 1.0 / kappa
    y = ar_observation(spread, phi)
    names = ("mu_prior", "pmu_prior", "sig2_prior", "psig_prior", "innovation", "innovation_var",
             "gain_mu", "gain_sig2", "mu_post", "posvar_mu", "sig2_post", "posvar_sig2")
    out = {name: np.full(len(spread), np.nan) for name in names}
    mu, P_mu, sig2, P_sig = params["mu0"], params["pmu0"], params["sig2_0"], params["psig0"]
    for t in range(len(phi), len(spread)):
        mu_prior, P_mu_prior = mu, P_mu + params["Q_mu"]          # predecir μ (A = 1)
        sig2_prior, P_sig_prior = sig2, P_sig + params["Q_sig2"]  # predecir σ² (A = 1)
        innovation, S, K_mu, mu, P_mu = kalman_update(mu_prior, P_mu_prior, y[t], C_mu, sig2_prior / kappa)
        u = innovation ** 2 - C_mu ** 2 * P_mu_prior
        _, _, K_sig, sig2, P_sig = kalman_update(sig2_prior, P_sig_prior, u, C_sig, params["R_sig2"])
        values = (mu_prior, P_mu_prior, sig2_prior, P_sig_prior, innovation, S, K_mu, K_sig, mu, P_mu, sig2, P_sig)
        for name, value in zip(names, values):
            out[name][t] = value
    return out


def log_likelihood(result: dict) -> float:
    """Log-verosimilitud gaussiana de las innovaciones del filtro de la media."""
    innovation, S = result["innovation"], result["innovation_var"]
    valid = ~np.isnan(innovation)
    return float(-0.5 * np.sum(np.log(2 * np.pi * S[valid]) + innovation[valid] ** 2 / S[valid]))


def base_params(spread_train: pd.Series) -> dict:
    """φ, κ, R_σ y valores iniciales a partir de train (sin Q todavía)."""
    ar = fit_ar(spread_train)
    kappa = variance_ratio(ar["phi"])
    sig2_0 = kappa * ar["sigma2_eps"]
    return {"ar_order": ar["order"], "ar_bic_order": ar["bic_order"], "phi": ar["phi"].tolist(),
            "ar_bic_by_order": ar["bic_by_order"],
            "sigma2_eps": ar["sigma2_eps"], "kappa": kappa, "R_sig2": float(np.var(ar["residuals"] ** 2)),
            "mu0": 0.0, "pmu0": float(spread_train.var()), "sig2_0": sig2_0, "psig0": sig2_0 ** 2,
            "spread_var_train": float(spread_train.var())}


def with_noise(params: dict, q_mu: float, q_sig2: float) -> dict:
    """Agrega Q_μ = q_μ·Var(S_train) y Q_σ = q_σ·(σ_0²)² a los parámetros base."""
    return {**params, "q_mu": q_mu, "q_sig2": q_sig2, "Q_mu": q_mu * params["spread_var_train"],
            "Q_sig2": q_sig2 * params["sig2_0"] ** 2}


def likelihood_grid(spread_train: np.ndarray, params: dict) -> pd.DataFrame:
    """Log-verosimilitud de train para cada par (q_μ, q_σ) de la rejilla."""
    return pd.DataFrame([{"q_mu": q_mu, "q_sig2": q_sig2,
                          "log_likelihood": log_likelihood(run_filter(spread_train, with_noise(params, q_mu, q_sig2)))}
                         for q_mu in Q_GRID for q_sig2 in Q_GRID])


def steady_state_posvar(Q: float, C: float, R: float) -> float:
    """P_{t|t} de equilibrio de un filtro escalar con A = 1 y C, Q, R constantes (ecuación de Riccati)."""
    prior = (Q + np.sqrt(Q ** 2 + 4 * Q * R / C ** 2)) / 2
    return float(prior - Q)


def convergence_day(posvar: np.ndarray, steady: float) -> int:
    """Primer día desde el cual P_{t|t} se queda a menos de SIGMA_CONVERGENCE_TOL (relativo) de su equilibrio."""
    if steady <= 0:
        return 0
    inside = np.abs(posvar / steady - 1) < SIGMA_CONVERGENCE_TOL
    stays = np.flip(np.cumprod(np.flip(inside))).astype(bool)
    return int(np.argmax(stays))


def calibrate(spread_train: pd.Series) -> tuple[dict, pd.DataFrame]:
    """Calibra todo con train: AR(p), R_σ, valores iniciales, (q_μ, q_σ) por máxima verosimilitud y burn-in.

    Burn-in = el mayor de dos días: P^μ dentro de su banda estable (criterio del filtro 1) y P^σ a menos de 1% de su
    equilibrio de Riccati. P^σ no depende de los datos (C_σ, Q_σ y R_σ son constantes) y converge a un valor fijo, así
    que la banda mediana ± 3·MAD del filtro 1 no le sirve: su MAD es casi cero y el corte dependería del redondeo.
    """
    params = base_params(spread_train)
    grid = likelihood_grid(spread_train.to_numpy(), params)
    best = grid.loc[grid["log_likelihood"].idxmax()]
    params = with_noise(params, float(best["q_mu"]), float(best["q_sig2"]))
    result = run_filter(spread_train.to_numpy(), params)
    order = params["ar_order"]
    steady = steady_state_posvar(params["Q_sig2"], 1.0 / params["kappa"], params["R_sig2"])
    mu_days = measure_burn_in(result["posvar_mu"][order:, None])
    sig2_days = convergence_day(result["posvar_sig2"][order:], steady)
    params.update(posvar_sig2_steady_state=steady, burn_in_mu_days=order + mu_days,
                  burn_in_sig2_days=order + sig2_days, burn_in_days=order + max(mu_days, sig2_days))
    return params, grid


def grid_summary(grid: pd.DataFrame, params: dict) -> dict:
    """Posición del óptimo en la rejilla y región a menos de 1.92 unidades de log-verosimilitud (IC 95% por RV)."""
    near = grid[grid["log_likelihood"] >= grid["log_likelihood"].max() - 1.92]
    positive = Q_GRID[Q_GRID > 0]
    return {
        "q_grid": "0 y 33 valores log-espaciados de 1e-8 a 1 (4 por década), para q_mu y q_sig2",
        "configurations": len(grid),
        "q_mu_on_edge": params["q_mu"] in (0.0, positive.min(), positive.max()),
        "q_sig2_on_edge": params["q_sig2"] in (0.0, positive.min(), positive.max()),
        "q_mu_within_1.92": [float(near["q_mu"].min()), float(near["q_mu"].max())],
        "q_sig2_within_1.92": [float(near["q_sig2"].min()), float(near["q_sig2"].max())],
        "log_likelihood_max": float(grid["log_likelihood"].max()),
    }


def build_frames(result: dict, spread: pd.DataFrame, start: int, burn_in_end: int) -> dict:
    """gain_2, posvar_2 y z_t con fecha, period y burn_in; NaN antes de que arranque el filtro 2."""
    def full(values: np.ndarray) -> np.ndarray:
        series = np.full(len(spread), np.nan)
        series[start:] = values
        return series

    mu_prev, sigma_prev = full(result["mu_prior"]), np.sqrt(full(result["sig2_prior"]))
    innovation = full(result["innovation"])
    frames = {
        "gain_2": pd.DataFrame({"mu": full(result["gain_mu"]), "sigma2": full(result["gain_sig2"])}),
        "posvar_2": pd.DataFrame({"mu": full(result["posvar_mu"]), "sigma2": full(result["posvar_sig2"])}),
        "z_t": pd.DataFrame({"z": prior_zscore(spread["spread"].to_numpy(), mu_prev, sigma_prev),
                             "mu_prev": mu_prev, "sigma_prev": sigma_prev, "spread": spread["spread"].to_numpy(),
                             "innovation_normalized": innovation / np.sqrt(full(result["innovation_var"]))}),
    }
    for frame in frames.values():
        frame.index = spread.index
        frame["period"] = spread["period"].to_numpy()
        frame["burn_in"] = np.arange(len(frame)) < burn_in_end
    return frames


def innovation_report(z_t: pd.DataFrame) -> dict:
    """Varianza, autocorrelación de orden 1 y Ljung-Box de las innovaciones normalizadas, por tramo (sin burn-in)."""
    report = {}
    for period in PERIODS:
        normalized = z_t.loc[(z_t["period"] == period) & ~z_t["burn_in"], "innovation_normalized"].to_numpy()
        report[period] = {"n": len(normalized), **innovation_diagnostics(normalized, np.ones_like(normalized))}
    return report


def run() -> None:
    """Calibra en train, corre el filtro 2 en toda la muestra con parámetros congelados y guarda los artefactos."""
    spread = pd.read_csv(SPREAD_PATH, index_col="date", parse_dates=True)
    start = int(np.argmax(~spread["burn_in"].to_numpy()))
    window = spread.iloc[start:]
    spread_train = window.loc[window["period"] == "train", "spread"]
    params, grid = calibrate(spread_train)
    result = run_filter(window["spread"].to_numpy(), params)
    if np.nanmin(result["sig2_post"]) <= 0:
        raise RuntimeError("La varianza latente del filtro 2 dejó de ser positiva")
    burn_in_end = start + params["burn_in_days"]
    frames = build_frames(result, spread, start, burn_in_end)
    diagnostics = innovation_report(frames["z_t"])
    summary = {
        "period_used": "train (después del burn-in del filtro 1)",
        "filter_start": spread.index[start].strftime("%Y-%m-%d"),
        "burn_in_end": spread.index[burn_in_end].strftime("%Y-%m-%d"),
        "train_days_used": len(spread_train),
        **params,
        "C_mu": 1.0 - float(np.sum(params["phi"])), "C_sig2": 1.0 / params["kappa"],
        "grid": grid_summary(grid, params),
        "innovations": diagnostics,
        "fallback_count": 0,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(OUT_DIR / f"{name}.csv", date_format="%Y-%m-%d")
    grid.to_csv(OUT_DIR / "filter2_q_grid.csv", index=False)
    (OUT_DIR / "filter2_params.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    train = diagnostics["train"]
    print(f"  AR({params['ar_order']}) phi = {np.round(params['phi'], 4).tolist()}, kappa = {params['kappa']:.2f}; "
          f"q_mu = {params['q_mu']:.2e}, q_sig2 = {params['q_sig2']:.2e} (orilla: {summary['grid']['q_mu_on_edge']}, "
          f"{summary['grid']['q_sig2_on_edge']}); burn-in hasta {summary['burn_in_end']}")
    print(f"  innovaciones normalizadas (train): varianza = {train['innovation_norm_var']:.3f}, "
          f"acf1 = {train['innovation_norm_acf1']:.3f}, Ljung-Box p = {train['ljung_box_pvalue']:.3f}")
