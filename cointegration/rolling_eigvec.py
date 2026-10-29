"""Primer eigenvector de una prueba de Johansen móvil, calculado en cada fecha con SOLO datos pasados.

Dueño: Milca
Lee: log-precios del grupo elegido (toda la muestra) y el β elegido en train.
Escribe: artifacts/cointegration/eigvec_rolling.csv (alimenta la figura 4).
Reglas: Cada fecha usa solo los WINDOW días que terminan en ella (sin look-ahead). No se usa para elegir nada.

El eigenvector se escala a longitud 1 y su signo se alinea con el β de train (producto punto >= 0), para que
no salte de signo entre fechas. Columnas extra: `similarity` (coseno con el β de train) y `rank_trace`.
"""
import numpy as np
import pandas as pd

from cointegration.johansen import johansen_test

WINDOW = 504


def rolling_first_eigenvector(log_prices: pd.DataFrame, var_order: int, reference: np.ndarray) -> pd.DataFrame:
    """Primer eigenvector unitario y alineado, similitud y r̂ en cada fecha desde el día WINDOW."""
    unit_reference = reference / np.linalg.norm(reference)
    rows = {}
    for end in range(WINDOW, len(log_prices) + 1):
        window = log_prices.iloc[end - WINDOW:end]
        johansen = johansen_test(window, var_order)
        vector = johansen["eigenvectors"][:, 0] / np.linalg.norm(johansen["eigenvectors"][:, 0])
        vector = vector if vector @ unit_reference >= 0 else -vector
        rows[window.index[-1]] = [*vector, float(vector @ unit_reference), johansen["rank_trace"]]
    frame = pd.DataFrame.from_dict(rows, orient="index", columns=[*log_prices.columns, "similarity", "rank_trace"])
    frame.index.name = "date"
    return frame
