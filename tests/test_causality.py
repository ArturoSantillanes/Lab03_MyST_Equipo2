"""Prueba obligatoria 1 (Arturo): el z-score en t no cambia al recalcular el pipeline con los datos hasta t.

Se recalcula la cadena completa (log-precios → filtro 1 → spread → filtro 2 → z) usando SOLO los precios hasta t, con
los parámetros congelados en train, para una fecha de train, una de test y una de valid: el z hasta t debe ser
idéntico al de la corrida con toda la muestra, y coincidir con el z_t entregado. Para t en test se recalibra además el filtro 2 desde cero con los datos hasta t: como la calibración solo
lee train, los parámetros y el z_t salen idénticos. También se verifica que z use los estadísticos PREVIOS.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalman.filter1_weights import P0_DIFFUSE, kalman_filter, observation_design
from kalman.filter2_spread import calibrate, run_filter

ROOT = Path(__file__).resolve().parent.parent
KALMAN_DIR = ROOT / "artifacts" / "kalman"


def load(name: str) -> dict:
    """JSON de artifacts/kalman."""
    path = KALMAN_DIR / name
    if not path.exists():
        pytest.skip("correr python main.py primero")
    return json.loads(path.read_text(encoding="utf-8"))


def delivered_z() -> pd.DataFrame:
    """z_t.csv entregado."""
    return pd.read_csv(KALMAN_DIR / "z_t.csv", index_col="date", parse_dates=True)


def spread_up_to(date: pd.Timestamp, filter1: dict) -> pd.Series:
    """Filtro 1 con parámetros congelados, corrido SOLO con precios hasta `date`; devuelve S_t."""
    prices = pd.read_csv(ROOT / "data" / "processed" / "prices.csv", index_col="date", parse_dates=True)
    log_prices = np.log(prices.loc[:date, filter1["tickers"]])
    centered = log_prices - pd.Series(filter1["train_mean_log_price"])[filter1["tickers"]]
    y, C = observation_design(centered, filter1["normalization_leg"])
    others = [t for t in filter1["tickers"] if t != filter1["normalization_leg"]]
    Q = np.diag([filter1["Q_diag"][t] for t in others])
    z0 = np.array([filter1["z0"][t] for t in others])
    result = kalman_filter(y, C, Q, filter1["R"], z0, P0_DIFFUSE * np.eye(len(others)))
    return pd.Series(result["innovation"], index=centered.index)


def z_up_to(date: pd.Timestamp, filter1: dict, filter2: dict) -> pd.Series:
    """Serie z recalculada con la cadena completa usando SOLO los datos hasta `date`."""
    spread = spread_up_to(date, filter1)
    start = filter1["burn_in_days"]
    result = run_filter(spread.to_numpy()[start:], filter2)
    z = (spread.to_numpy()[start:] - result["mu_prior"]) / np.sqrt(result["sig2_prior"])
    return pd.Series(z, index=spread.index[start:])


def test_zscore_at_t_does_not_change_when_recomputed_with_data_up_to_t():
    # (a) causalidad: con datos hasta t, el z de t (y de todos los días previos) es idéntico bit a bit al de la
    #     corrida con toda la muestra. (b) reproducción: coincide con el z_t.csv entregado; la tolerancia 1e-6 solo
    #     cubre el redondeo de punto flotante entre computadoras (los artefactos pueden venir de otra máquina).
    filter1, filter2 = load("filter1_params.json"), load("filter2_params.json")
    z_t = delivered_z()
    full = z_up_to(z_t.index[-1], filter1, filter2)
    tradable = z_t[~z_t["burn_in"]]
    dates = [tradable[tradable["period"] == p].index[len(tradable[tradable["period"] == p]) // 2]
             for p in ("train", "test", "valid")]
    for date in dates:
        truncated = z_up_to(date, filter1, filter2)
        np.testing.assert_array_equal(truncated.to_numpy(), full.loc[:date].to_numpy())
        assert truncated.iloc[-1] == pytest.approx(z_t.at[date, "z"], abs=1e-6), date


def test_recalibrating_filter2_with_data_up_to_t_gives_the_same_zscore():
    filter2 = load("filter2_params.json")
    z_t = delivered_z()
    test_dates = z_t.index[z_t["period"] == "test"]
    date = test_dates[len(test_dates) // 2]
    window = z_t.loc[filter2["filter_start"]:date]
    params, _ = calibrate(window.loc[window["period"] == "train", "spread"])
    result = run_filter(window["spread"].to_numpy(), params)
    z_last = (window["spread"].iloc[-1] - result["mu_prior"][-1]) / np.sqrt(result["sig2_prior"][-1])
    assert params["q_mu"] == filter2["q_mu"] and params["q_sig2"] == filter2["q_sig2"]
    assert z_last == pytest.approx(z_t.at[date, "z"], rel=1e-9)


def test_zscore_uses_prior_not_posterior_statistics():
    filter2 = load("filter2_params.json")
    z_t = delivered_z()
    window = z_t.loc[filter2["filter_start"]:]
    result = run_filter(window["spread"].to_numpy(), filter2)
    order = filter2["ar_order"]
    np.testing.assert_allclose(result["mu_prior"][order + 1:], result["mu_post"][order:-1])
    np.testing.assert_allclose(window["z"], (window["spread"] - window["mu_prev"]) / window["sigma_prev"])
    z_prior = window["z"].to_numpy()[order:]
    z_posterior = (window["spread"].to_numpy() - result["mu_post"])[order:] / np.sqrt(result["sig2_post"][order:])
    assert np.mean(np.abs(z_posterior)) < np.mean(np.abs(z_prior))


def test_signal_zscore_from_vecm_residual_matches_filter2_zscore():
    path = ROOT / "artifacts" / "strategy" / "signal_zscore.csv"
    if not path.exists():
        pytest.skip("correr python main.py primero")
    signal = pd.read_csv(path, index_col="date", parse_dates=True)
    np.testing.assert_allclose(signal["z"], delivered_z()["z"], rtol=1e-9, atol=1e-12)
