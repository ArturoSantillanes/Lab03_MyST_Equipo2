"""Etapa 'backtest': corre el motor con costos sobre train, test y valid.

Dueño: Paula
Lee: data/processed/prices.csv, artifacts/kalman/*.csv, artifacts/strategy/params_default.json
Escribe: artifacts/backtest/equity.csv, trades.csv, costs.csv, params.json, grid_search.csv
Reglas: Capital inicial $1,000,000; comisión 0.125% por pata en entrada y salida; préstamo 0.25% anual con devengo diario. Umbrales solo con train.
"""


def run() -> None:
    """Ejecuta esta etapa del pipeline (pendiente de implementar)."""
    raise NotImplementedError("Lo implementa Paula")
