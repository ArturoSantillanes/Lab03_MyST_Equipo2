"""Valida y limpia los precios: faltantes, duplicados, saltos anómalos y fechas desalineadas.

Dueño: Milca
Lee: data/raw/prices_raw.csv
Escribe: (en memoria) precios limpios y un reporte con el conteo de cada caso.
Reglas: Reportar cuántos datos se encontraron y cómo se resolvió cada caso. No inventar precios.

Reglas de resolución:
    fechas duplicadas      -> se conserva la primera aparición y se descartan las demás
    precios no positivos   -> se tratan como faltantes (son errores de datos)
    fechas desalineadas    -> fechas en que algún activo no tiene precio y otros sí: se descarta la fecha
                              completa (no se rellena: rellenar inventa precios)
    faltantes              -> mismo tratamiento: la fecha se descarta para todos los activos
    saltos anómalos        -> |log-retorno diario| > JUMP_THRESHOLD se marcan; si el día siguiente revierte
                              más de la mitad del salto se marca como "posible error". No se modifican: el precio
                              ya viene ajustado por splits y dividendos, y corregir a mano sería manipular datos.
"""
import numpy as np
import pandas as pd

JUMP_THRESHOLD = 0.40
REVERSAL_SHARE = 0.5
MIN_YEARS, MAX_YEARS = 10, 15


def standardize(raw: pd.DataFrame) -> pd.DataFrame:
    """Nombres en minúsculas y fechas como índice ordenado."""
    prices = raw.rename(columns=lambda c: str(c).strip().lower()).copy()
    prices.index = pd.DatetimeIndex(prices.index).normalize()
    prices.index.name = "date"
    return prices.sort_index()


def drop_duplicate_dates(prices: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Quita fechas repetidas (conserva la primera) y las cuenta."""
    duplicated = prices.index.duplicated(keep="first")
    report["duplicate_dates"] = int(duplicated.sum())
    return prices[~duplicated]


def drop_misaligned_dates(prices: pd.DataFrame, report: dict) -> pd.DataFrame:
    """Cuenta faltantes y precios no positivos, y descarta las fechas en que algún activo no tiene precio válido."""
    non_positive = prices <= 0
    report["non_positive_prices"] = int(non_positive.sum().sum())
    prices = prices.mask(non_positive)
    missing = prices.isna()
    report["missing_values_by_ticker"] = {t: int(n) for t, n in missing.sum().items()}
    partial = missing.any(axis=1) & ~missing.all(axis=1)
    report["misaligned_dates"] = int(partial.sum())
    report["dates_without_any_price"] = int(missing.all(axis=1).sum())
    return prices[~missing.any(axis=1)]


def flag_jumps(prices: pd.DataFrame, report: dict) -> None:
    """Marca (sin modificar) los saltos anómalos y cuáles parecen error porque revierten al día siguiente."""
    log_returns = np.log(prices).diff()
    flags = []
    for date, ticker in (log_returns.abs() > JUMP_THRESHOLD).stack().loc[lambda s: s].index:
        jump = float(log_returns.at[date, ticker])
        position = log_returns.index.get_loc(date)
        following = float(log_returns[ticker].iloc[position + 1]) if position + 1 < len(log_returns) else 0.0
        flags.append({"date": date.strftime("%Y-%m-%d"), "ticker": ticker, "log_return": round(jump, 4),
                      "next_day_log_return": round(following, 4),
                      "possible_error": bool(np.sign(following) == -np.sign(jump) and abs(following) > REVERSAL_SHARE * abs(jump))})
    report["anomalous_jumps"] = flags
    report["anomalous_jumps_count"] = len(flags)
    report["possible_data_errors_count"] = sum(f["possible_error"] for f in flags)


def validate(prices: pd.DataFrame) -> None:
    """Verifica que el resultado sea usable; lanza ValueError si no."""
    if prices.isna().any().any() or (prices <= 0).any().any():
        raise ValueError("Quedan precios faltantes o no positivos")
    if not prices.index.is_monotonic_increasing or prices.index.has_duplicates:
        raise ValueError("Las fechas no son estrictamente crecientes")
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    if not MIN_YEARS - 0.05 <= years <= MAX_YEARS + 0.05:
        raise ValueError(f"La muestra cubre {years:.2f} años; se piden de {MIN_YEARS} a {MAX_YEARS}")


def clean_prices(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aplica las reglas en orden y devuelve (precios limpios, reporte de conteos)."""
    report = {"rows_raw": len(raw), "tickers": len(raw.columns)}
    prices = drop_misaligned_dates(drop_duplicate_dates(standardize(raw), report), report)
    flag_jumps(prices, report)
    validate(prices)
    report.update(rows_clean=len(prices), start=prices.index[0].strftime("%Y-%m-%d"),
                  end=prices.index[-1].strftime("%Y-%m-%d"),
                  years=round((prices.index[-1] - prices.index[0]).days / 365.25, 2))
    return prices, report
