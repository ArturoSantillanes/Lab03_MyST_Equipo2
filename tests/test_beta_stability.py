"""Prueba 3 de Milca (obligatoria): el vector de pesos mantiene escala y signo estables entre reestimaciones.

Cada día el filtro de Kalman 1 reestima β̂_t. Sobre los pesos entregados (artifacts/kalman/beta_t.csv), en train
después del burn-in, se exige:
    escala: la pata de normalización vale exactamente +1 en todas las fechas
    signo:  ningún peso cambia de signo entre dos reestimaciones consecutivas (criterio 6.8.4 c, sin umbral)
Además, con datos sintéticos, agregar un día no modifica los pesos anteriores (reestimación consecutiva causal).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kalman.filter1_weights import kalman_filter

WEIGHTS_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "kalman" / "beta_t.csv"


def delivered_train_weights() -> pd.DataFrame:
    """Pesos entregados en train, después del burn-in."""
    if not WEIGHTS_PATH.exists():
        pytest.skip("correr python main.py primero")
    weights = pd.read_csv(WEIGHTS_PATH, index_col="date", parse_dates=True, encoding="utf-8")
    return weights[(weights["period"] == "train") & ~weights["burn_in"]].drop(columns=["period", "burn_in"])


def test_scale_is_fixed_by_the_normalization_leg():
    weights = delivered_train_weights()
    assert int((weights == 1.0).all().sum()) == 1


def test_no_weight_changes_sign_between_consecutive_reestimations():
    weights = delivered_train_weights()
    sign_changes = (np.sign(weights).diff().abs() > 0).sum()
    assert int(sign_changes.sum()) == 0, f"cambios de signo por pata en train: {sign_changes.to_dict()}"


def test_adding_one_day_does_not_change_past_weights():
    rng = np.random.default_rng(42)
    C = np.cumsum(rng.normal(0, 0.01, (600, 2)), axis=0)
    y = C @ np.array([0.6, -0.3]) + rng.normal(0, 0.005, 600)
    Q, R, z0, P0 = np.eye(2) * 1e-7, 2.5e-5, np.zeros(2), np.eye(2) * 1e4
    shorter, longer = kalman_filter(y[:500], C[:500], Q, R, z0, P0), kalman_filter(y[:501], C[:501], Q, R, z0, P0)
    np.testing.assert_array_equal(shorter["z_prior"], longer["z_prior"][:500])
