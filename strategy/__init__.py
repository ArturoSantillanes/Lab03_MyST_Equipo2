"""Etapa 'strategy': VECM, z-score de señales, máquina de estados y parámetros por defecto.

Dueño: Arturo (strategy/sizing.py es de Paula)
Lee: data/processed/prices.csv, data/processed/splits.json, artifacts/cointegration/selected_group.json,
     artifacts/kalman/beta_t.csv, spread_t.csv, filter1_params.json, gain_1.csv, gain_2.csv, z_t.csv
Escribe: artifacts/strategy/params_default.json, vecm_summary.json, signal_zscore.csv, positions_default.csv,
         direction_check.json; artifacts/figures/fig2_spread_senales.png, fig5_ganancia_kalman.png, fig6_zscore.png
Reglas: el VECM se estima SOLO con train; los umbrales definitivos los calibra Paula SOLO con train usando
generate_positions. Sin look-ahead. Expone generate_positions(z, params).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data.split import slice_period
from strategy.figures import plot_kalman_gains, plot_spread_signals, plot_zscore
from strategy.signals import default_params, direction_check, generate_positions, spread_leg_returns
from strategy.vecm import equilibrium_error, fit_vecm, free_johansen_beta, lag_order
from strategy.zscore import signal_zscore

__all__ = ["generate_positions", "run"]

ROOT = Path(__file__).resolve().parent.parent
KALMAN_DIR = ROOT / "artifacts" / "kalman"
OUT_DIR = ROOT / "artifacts" / "strategy"


def read_json(path: Path) -> dict:
    """Lee un JSON en UTF-8."""
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, content: dict) -> None:
    """Escribe un JSON en UTF-8 con sangría."""
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def read_dated(name: str) -> pd.DataFrame:
    """CSV de artifacts/kalman con la fecha como índice."""
    return pd.read_csv(KALMAN_DIR / name, index_col="date", parse_dates=True)


def vecm_summary(log_prices: pd.DataFrame, selection: dict, splits: dict, ect: pd.Series, spread: pd.Series,
                 start: pd.Timestamp) -> dict:
    """Estima el VECM en train con el β de Johansen y arma el resumen (incluye controles de consistencia)."""
    train = slice_period(log_prices, splits, "train")
    beta = pd.Series(selection["beta"])[log_prices.columns]
    orders = lag_order(train)
    k_ar_diff = selection["var_order"]["used"] - 1
    fit = fit_vecm(train, beta, k_ar_diff)
    free_beta = free_johansen_beta(train, k_ar_diff, selection["normalization_leg"])
    projected = fit["residuals"] @ beta
    ect_train = train @ beta
    after_burn_in = ect_train.index >= start
    return {
        "period_used": "train", "specification": "p = 1 (k_ar_diff = 0), r = 1, constante sin restringir (caso 3)",
        "k_ar_diff": k_ar_diff, "select_order_k_ar_diff_train": orders, "n_obs": fit["n_obs"],
        "beta_johansen_fixed": beta.to_dict(), "alpha": fit["alpha"].to_dict(),
        "alpha_tvalues": fit["alpha_tvalues"].to_dict(), "const": fit["const"].to_dict(), "gamma": fit["gamma"],
        "r_squared": fit["r_squared"].to_dict(), "sigma_u": fit["sigma_u"].to_dict(),
        "ect_speed_beta_alpha": fit["ect_speed"], "ect_half_life_days": fit["ect_half_life_days"],
        "free_vecm_beta_normalized": free_beta.to_dict(),
        "free_vs_fixed_beta_max_abs_diff": float((free_beta - beta).abs().max()),
        "residual_options_train": {
            "ect_static_beta_acf1": float(ect_train.autocorr()),
            "beta_projected_innovations_acf1": float(projected.autocorr()),
            "corr_ect_dynamic_vs_static_train_after_burn_in": float(
                np.corrcoef(ect.loc[ect_train.index[after_burn_in]], ect_train[after_burn_in])[0, 1]),
        },
        "ect_dynamic_vs_filter1_spread_max_abs_diff": float((ect - spread).abs().max()),
    }


def run() -> None:
    """VECM, z-score de señales, posiciones con los parámetros por defecto, verificación de dirección y figuras."""
    selection = read_json(ROOT / "artifacts" / "cointegration" / "selected_group.json")
    splits = read_json(ROOT / "data" / "processed" / "splits.json")
    filter1 = read_json(KALMAN_DIR / "filter1_params.json")
    tickers = selection["tickers"]
    prices = pd.read_csv(ROOT / "data" / "processed" / "prices.csv", index_col="date", parse_dates=True)
    log_prices = np.log(prices[tickers])
    beta_t = read_dated("beta_t.csv")[tickers]
    spread, z_t = read_dated("spread_t.csv"), read_dated("z_t.csv")

    ect = equilibrium_error(log_prices, beta_t, pd.Series(filter1["train_mean_log_price"]))
    signal = signal_zscore(ect, z_t)
    start = spread.index[int(np.argmax(~spread["burn_in"].to_numpy()))]
    summary = vecm_summary(log_prices, selection, splits, ect, spread["spread"], start)
    params = default_params(selection["half_life_days"])
    tradable_z = signal["z"].where(~signal["burn_in"])
    positions = generate_positions(tradable_z, params)
    check = direction_check(positions, spread_leg_returns(log_prices, beta_t), tradable_z, signal["period"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUT_DIR / "params_default.json", {
        **params, "period_used": "train (vida media de Johansen de train, selected_group.json)",
        "note": "Valores de arranque, no óptimos. Los definitivos los elige Paula con grid search SOLO en train "
                "(con número mínimo de operaciones) usando strategy.generate_positions."})
    write_json(OUT_DIR / "vecm_summary.json", summary)
    write_json(OUT_DIR / "direction_check.json", check)
    signal.to_csv(OUT_DIR / "signal_zscore.csv", date_format="%Y-%m-%d")
    positions.assign(period=signal["period"]).to_csv(OUT_DIR / "positions_default.csv", date_format="%Y-%m-%d")

    plot_spread_signals(z_t, positions, params, splits, start)
    plot_kalman_gains(read_dated("gain_1.csv"), read_dated("gain_2.csv"), tickers, splits)
    plot_zscore(z_t, params, splits, start)

    print(f"  VECM (train): beta'alpha = {summary['ect_speed_beta_alpha']:.4f}, vida media implícita = "
          f"{summary['ect_half_life_days']:.1f} días; select_order k_ar_diff = {summary['select_order_k_ar_diff_train']}")
    print(f"  parámetros por defecto: {params['z_entry']}/{params['z_exit']}/{params['z_stop']}, "
          f"max_hold = {params['max_hold_days']} días; dirección (train): corr(z, rendimiento del spread al día "
          f"siguiente) = {check['corr_z_vs_next_day_spread_return']:.3f}, {check['train_trades_with_positive_gross']} de "
          f"{check['train_trades']} operaciones con rendimiento bruto positivo")
