"""Z-score de señales: residual de equilibrio del VECM normalizado con las estimaciones previas del filtro 2.

Dueño: Arturo
Lee: (se lo pasa strategy.run) el ECT_t de strategy.vecm.equilibrium_error y artifacts/kalman/z_t.csv
Escribe: (lo guarda strategy.run) artifacts/strategy/signal_zscore.csv
Reglas: z_t = (ECT_t − μ̂_{t|t−1}) / σ̂_{t|t−1}, con la media y la varianza PREVIAS a la actualización del filtro 2
(misma fórmula que kalman.filter2_spread.prior_zscore). Sin look-ahead.

Supuesto (la frase del enunciado "residuales del VECM" admite varias lecturas; ver docs/secciones/arturo.md):
se usa el residual de EQUILIBRIO del VECM, ECT_t = β̂_tᵀ(P_t − μ_train), que es lo que el VECM corrige, y no las
innovaciones ε_t de cada ecuación, que son ruido casi blanco y no revierten (no hay nada que operar en ellas).
"""
import pandas as pd

from kalman.filter2_spread import prior_zscore


def signal_zscore(ect: pd.Series, filter2: pd.DataFrame) -> pd.DataFrame:
    """ECT_t normalizado con mu_prev y sigma_prev del filtro 2; conserva period y burn_in."""
    z = prior_zscore(ect, filter2["mu_prev"], filter2["sigma_prev"])
    return pd.DataFrame({"ect": ect, "z": z, "period": filter2["period"], "burn_in": filter2["burn_in"]})
