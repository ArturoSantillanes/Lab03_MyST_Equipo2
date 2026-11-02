"""Sharpe móvil de 252 días.

Dueño: Paula
Lee: Curva de capital.
Escribe: artifacts/backtest/rolling_sharpe.csv
Reglas: Ventana de 252 días sobre toda la muestra.

La ventana corre sobre los rendimientos diarios de los tres tramos encadenados en orden (train operable, test, valid).
Cada valor usa solo los 252 días que terminan en esa fecha y se etiqueta con el tramo de esa fecha; las primeras 251
fechas no tienen valor. Las ventanas que cruzan una frontera mezclan dos tramos, y así se indica en window_periods.
"""
import math

import pandas as pd

from performance.metrics import ANNUALIZATION, RISK_FREE_RATE, ZERO_TOL

WINDOW = 252


def rolling_sharpe(returns: pd.Series, window: int = WINDOW) -> pd.Series:
    """Sharpe anualizado móvil (media/desviación muestral·√252, tasa libre de riesgo 0)."""
    excess = returns - RISK_FREE_RATE / ANNUALIZATION
    mean = excess.rolling(window).mean()
    std = excess.rolling(window).std(ddof=1)
    return (mean / std * math.sqrt(ANNUALIZATION)).where(std > ZERO_TOL)


def rolling_table(equity: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Tabla para rolling_sharpe.csv: fecha, tramo, Sharpe móvil y tramos que abarca la ventana."""
    period_codes = equity["period"]
    first_period = period_codes.shift(window - 1)
    spans = first_period.where(first_period == period_codes, first_period + "+" + period_codes)
    return pd.DataFrame({"period": equity["period"], "rolling_sharpe_252": rolling_sharpe(equity["daily_return"], window),
                         "window_periods": spans.where(first_period.notna())}, index=equity.index)
