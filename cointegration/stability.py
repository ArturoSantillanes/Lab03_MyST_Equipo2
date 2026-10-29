"""Etapa 7 del embudo: estabilidad de la cointegración en sub-ventanas móviles de train (obligatoria).

Dueño: Milca
Lee: log-precios de train del grupo elegido.
Escribe: (en memoria) r̂, ADF y vida media por sub-ventana; la fracción de sub-ventanas estables entra en la
selección final del grupo (cointegration/final_selection.py) como criterio, no solo como dato.
Reglas: SOLO train.

Se repiten las etapas 4, 5 y 6 (mismo orden p, misma especificación) en ventanas de WINDOW = 504 días
(2 años: suficientes observaciones para un VECM de 6 series) que avanzan STEP = 126 días (medio año), lo que
da 14 sub-ventanas solapadas que cubren todo train. Una sub-ventana es "estable" si el grupo sigue cointegrado
(r̂ >= 1 por la traza) Y el spread de la mejor columna dentro de r̂ sigue pasando ADF (p < 0.05).
Si la cointegración se pierde fuera de la muestra, la estrategia pierde sentido: aquí se minimiza el error
dentro de la muestra y una ruptura posterior es posible; se documenta en lugar de esconderse.
"""
import pandas as pd

from cointegration.johansen import johansen_test
from cointegration.spread_checks import MAX_ADF_PVALUE, column_table

WINDOW = 504
STEP = 126


def stability_windows(log_prices: pd.DataFrame, var_order: int) -> pd.DataFrame:
    """r̂, columna elegida, ADF y vida media en cada sub-ventana de train."""
    rows = []
    for end in range(WINDOW, len(log_prices) + 1, STEP):
        window = log_prices.iloc[end - WINDOW:end]
        johansen = johansen_test(window, var_order)
        rank = johansen["rank_trace"]
        columns = column_table(window, johansen["eigenvectors"], rank)
        pool = columns[columns["within_rank"]] if rank >= 1 else columns
        best = pool.loc[pool["adf_stat"].idxmin()]
        passes_adf = bool(best["adf_pvalue"] < MAX_ADF_PVALUE)
        rows.append({"start": window.index[0].strftime("%Y-%m-%d"), "end": window.index[-1].strftime("%Y-%m-%d"),
                     "r_hat": rank, "best_adf_pvalue": best["adf_pvalue"], "best_half_life_days": best["half_life_days"],
                     "spread_passes_adf": passes_adf, "stable": rank >= 1 and passes_adf})
    return pd.DataFrame(rows)


def stability_share(windows: pd.DataFrame) -> float:
    """Fracción de sub-ventanas estables."""
    return float(windows["stable"].mean())
