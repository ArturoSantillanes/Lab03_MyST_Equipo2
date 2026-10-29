"""Etapa 7 del embudo: estabilidad de la cointegración en sub-ventanas móviles de train (obligatoria).

Dueño: Milca
Lee: log-precios de train del grupo elegido.
Escribe: (en memoria) r̂, ADF y vida media por sub-ventana, y la fracción de sub-ventanas cointegradas.
Reglas: SOLO train.

Se repiten las etapas 4, 5 y 6 (mismo orden p, misma especificación) en ventanas de WINDOW = 504 días
(2 años: suficientes observaciones para un VECM de 6 series) que avanzan STEP = 126 días (medio año), lo que
da 14 sub-ventanas solapadas que cubren todo train. Una sub-ventana "sigue cointegrada" si r̂ >= 1 y alguna
columna dentro de r̂ tiene ADF p < 0.05 y vida media entre 5 y 60 días.
Si la cointegración se pierde fuera de la muestra, la estrategia pierde sentido: aquí se minimiza el error
dentro de la muestra y una ruptura posterior es posible; se documenta en lugar de esconderse.
"""
import pandas as pd

from cointegration.johansen import johansen_test
from cointegration.spread_checks import choose_column, column_table

WINDOW = 504
STEP = 126


def stability_windows(log_prices: pd.DataFrame, var_order: int) -> pd.DataFrame:
    """r̂, columna elegida, ADF y vida media en cada sub-ventana de train."""
    rows = []
    for end in range(WINDOW, len(log_prices) + 1, STEP):
        window = log_prices.iloc[end - WINDOW:end]
        johansen = johansen_test(window, var_order)
        columns = column_table(window, johansen["eigenvectors"], johansen["rank_trace"])
        chosen = choose_column(columns)
        best = columns.iloc[chosen if chosen is not None else int(columns["adf_stat"].idxmin())]
        rows.append({"start": window.index[0].strftime("%Y-%m-%d"), "end": window.index[-1].strftime("%Y-%m-%d"),
                     "r_hat": johansen["rank_trace"], "best_adf_pvalue": best["adf_pvalue"],
                     "best_half_life_days": best["half_life_days"], "spread_passes_adf": bool(best["adf_pvalue"] < 0.05),
                     "still_cointegrated": chosen is not None and johansen["rank_trace"] >= 1})
    return pd.DataFrame(rows)
