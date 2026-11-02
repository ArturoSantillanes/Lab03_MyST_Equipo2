"""Prueba de Arturo: el filtro 2 calculado a mano con las fórmulas del profesor coincide con el código.

(1) Un paso con números sintéticos. (2) Los primeros cinco periodos reales del filtro 2, recalculados con aritmética
escalar (sin llamar al filtro) y comparados con los artefactos entregados.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalman.filter2_spread import run_filter, variance_ratio

KALMAN_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "kalman"


def hand_step(state: dict, spread_now: float, spread_lags: list[float], p: dict) -> dict:
    """Un periodo del filtro 2 escrito fórmula por fórmula (A = 1)."""
    mu_prior = state["mu"]                                          # μ̂_{t|t−1} = μ̂_{t−1|t−1}
    pmu_prior = state["pmu"] + p["Q_mu"]                            # P^μ_{t|t−1} = P^μ_{t−1|t−1} + Q_μ
    sig2_prior = state["sig2"]                                      # σ̂²_{t|t−1} = σ̂²_{t−1|t−1}
    psig_prior = state["psig"] + p["Q_sig2"]                        # P^σ_{t|t−1} = P^σ_{t−1|t−1} + Q_σ
    z = (spread_now - mu_prior) / np.sqrt(sig2_prior)               # z_t con estadísticos PREVIOS
    c_mu = 1.0 - sum(p["phi"])                                      # C_μ = 1 − Σφ_i
    y = spread_now - sum(f * s for f, s in zip(p["phi"], spread_lags))  # y_t = S_t − Σφ_i·S_{t−i}
    innovation = y - c_mu * mu_prior                                # ỹ_t = y_t − C_μ·μ̂_{t|t−1}
    s_mu = c_mu * pmu_prior * c_mu + sig2_prior / p["kappa"]       # S_t = C_μ·P·C_μ + R_t,  R_t = σ̂²_{t|t−1}/κ
    k_mu = pmu_prior * c_mu / s_mu                                  # K_t = P·C_μ / S_t
    mu_post = mu_prior + k_mu * innovation                          # μ̂_{t|t}
    pmu_post = (1 - k_mu * c_mu) * pmu_prior                        # P^μ_{t|t}
    u = innovation ** 2 - c_mu ** 2 * pmu_prior                     # u_t = ỹ_t² − C_μ²·P^μ_{t|t−1}
    c_sig = 1.0 / p["kappa"]                                        # C_σ = 1/κ
    innovation_sig = u - c_sig * sig2_prior
    s_sig = c_sig * psig_prior * c_sig + p["R_sig2"]
    k_sig = psig_prior * c_sig / s_sig
    return {"mu": mu_post, "pmu": pmu_post, "sig2": sig2_prior + k_sig * innovation_sig,
            "psig": (1 - k_sig * c_sig) * psig_prior, "z": z, "mu_prior": mu_prior, "sig2_prior": sig2_prior,
            "k_mu": k_mu, "k_sig": k_sig, "innovation": innovation, "s_mu": s_mu}


def test_ar1_variance_ratio_matches_closed_form():
    assert variance_ratio(np.array([0.9])) == pytest.approx(1 / (1 - 0.81))


def test_one_synthetic_step_by_hand_matches_code():
    params = {"phi": [0.9], "kappa": 1 / (1 - 0.81), "Q_mu": 1e-6, "Q_sig2": 1e-9, "R_sig2": 1e-8,
              "mu0": 0.01, "pmu0": 4e-4, "sig2_0": 2e-4, "psig0": 4e-8}
    spread = np.array([0.020, 0.035])
    hand = hand_step({"mu": params["mu0"], "pmu": params["pmu0"], "sig2": params["sig2_0"], "psig": params["psig0"]},
                     spread[1], [spread[0]], params)
    code = run_filter(spread, params)
    assert code["mu_prior"][1] == pytest.approx(hand["mu_prior"])
    assert code["innovation"][1] == pytest.approx(hand["innovation"])
    assert code["innovation_var"][1] == pytest.approx(hand["s_mu"])
    assert code["gain_mu"][1] == pytest.approx(hand["k_mu"])
    assert code["mu_post"][1] == pytest.approx(hand["mu"])
    assert code["posvar_mu"][1] == pytest.approx(hand["pmu"])
    assert code["gain_sig2"][1] == pytest.approx(hand["k_sig"])
    assert code["sig2_post"][1] == pytest.approx(hand["sig2"])
    assert code["posvar_sig2"][1] == pytest.approx(hand["psig"])


def test_first_five_real_periods_by_hand_match_artifacts():
    if not (KALMAN_DIR / "filter2_params.json").exists():
        pytest.skip("correr python main.py primero")
    params = json.loads((KALMAN_DIR / "filter2_params.json").read_text(encoding="utf-8"))
    z_t = pd.read_csv(KALMAN_DIR / "z_t.csv", index_col="date", parse_dates=True)
    gain = pd.read_csv(KALMAN_DIR / "gain_2.csv", index_col="date", parse_dates=True)
    posvar = pd.read_csv(KALMAN_DIR / "posvar_2.csv", index_col="date", parse_dates=True)
    order = params["ar_order"]
    start = z_t.index.get_loc(pd.Timestamp(params["filter_start"]))
    spread = z_t["spread"].to_numpy()
    state = {"mu": params["mu0"], "pmu": params["pmu0"], "sig2": params["sig2_0"], "psig": params["psig0"]}
    for t in range(start + order, start + order + 5):
        lags = [spread[t - i] for i in range(1, order + 1)]
        state = hand_step(state, spread[t], lags, params)
        assert z_t["z"].iloc[t] == pytest.approx(state["z"], rel=1e-9)
        assert z_t["mu_prev"].iloc[t] == pytest.approx(state["mu_prior"], rel=1e-9)
        assert z_t["sigma_prev"].iloc[t] == pytest.approx(np.sqrt(state["sig2_prior"]), rel=1e-9)
        assert gain["mu"].iloc[t] == pytest.approx(state["k_mu"], rel=1e-9)
        assert gain["sigma2"].iloc[t] == pytest.approx(state["k_sig"], rel=1e-9)
        assert posvar["mu"].iloc[t] == pytest.approx(state["pmu"], rel=1e-9)
        assert posvar["sigma2"].iloc[t] == pytest.approx(state["psig"], rel=1e-9)
