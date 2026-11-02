"""Calibra umbrales (z_entry, z_exit, z_stop, tenencia máxima) con grid search.

Dueño: Paula
Lee: Señales de Arturo y el motor de backtest.
Escribe: artifacts/backtest/grid_search.csv, params.json
Reglas: SOLO train. Si el óptimo cae en el borde de la rejilla, decirlo.

Procedimiento (fijado antes de ver resultados):
1. Rejilla completa de z_entry × z_exit × z_stop × max_hold_days con 0 < z_exit < z_entry < z_stop (GRID). z_stop = inf
   es "sin stop" y es el extremo natural de esa dimensión, no un recorte.
2. Para cada configuración: strategy.generate_positions sobre el z operable de train (burn_in == False) y el motor con
   rebalanceo diario y costos completos. Objetivo = Sharpe anualizado de los rendimientos diarios netos de train.
3. Restricción de operaciones mínimas: con menos de N_MIN operaciones cerradas el objetivo vale PENALTY (−1e9). Sin
   esto el optimizador elige configuraciones con 2 o 3 operaciones ganadoras: un problema de tamaño de muestra, no una
   estrategia. Si ninguna configuración llega a N_MIN, NO se relaja en silencio: se informa el máximo alcanzable.
4. Meseta o pico: el vecindario de una configuración son las que difieren a lo más un paso de la rejilla en cada
   dimensión. El mejor es MESETA si la mediana del objetivo de su vecindario no queda más de UN ERROR ESTÁNDAR del
   Sharpe por debajo del suyo (EE de Lo, 2002: √((1 + SR²/2)/T) por día, anualizado; con T ≈ 1,161 días de train es
   ≈ 0.47). Dentro de un error estándar los Sharpe no se distinguen estadísticamente. Si no es meseta, es pico aislado
   y se elige el centro de la mejor meseta = la configuración viable con la mayor mediana de su vecindario (incluida
   ella misma).
5. Se reporta si cada parámetro elegido quedó en la orilla de su dimensión.
Historia (declarada; todo visto SOLO en train y antes de evaluar test o valid):
- La primera rejilla (z_entry desde 0.5, z_exit desde 0.25, max_hold_days hasta 60; 1,314 configuraciones) tenía el
  mejor en la orilla de max_hold_days y de z_exit: se amplió a z_exit 0.1 y max_hold_days hasta 150 (2,470).
- La configuración elegida cayó entonces en la orilla inferior de z_entry (0.5): se agregó z_entry = 0.25 (2,535
  configuraciones, la rejilla final) y la elección no cambió; ya no está en esa orilla.
- La primera versión de la regla de meseta pedía una mediana ≥ 0.75 × Sharpe del mejor, que no está definida con
  Sharpe negativo; se cambió por la del error estándar. Con ambas versiones la configuración elegida es la misma.
- max_hold_days = 150 sigue en la orilla, pero en train no se activa (la operación más larga dura menos), así que el
  objetivo de train es idéntico para cualquier valor mayor: la superficie es plana más allá de la orilla.
La banda de no-rebalanceo se elige después, también solo en train y con los umbrales ya congelados (choose_band).
"""
import itertools
import math

import numpy as np
import pandas as pd

from backtest.engine import run_backtest
from performance.metrics import ANNUALIZATION, sharpe_ratio
from strategy import generate_positions
from strategy.sizing import INITIAL_CAPITAL

N_MIN_DEFAULT = 30
PENALTY = -1e9
GRID = {
    "z_entry": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5],
    "z_exit": [0.1, 0.25, 0.5, 0.75, 1.0],
    "z_stop": [2.5, 3.0, 3.5, 4.0, np.inf],
    "max_hold_days": [5, 10, 15, 20, 25, 31, 40, 50, 60, 80, 100, 125, 150],
}
BAND_GRID = [0.0, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50, np.inf]
PARAMS = list(GRID)


def grid_configurations(grid: dict = GRID) -> list[dict]:
    """Todas las combinaciones válidas (0 < z_exit < z_entry < z_stop)."""
    combos = (dict(zip(grid, values)) for values in itertools.product(*grid.values()))
    return [c for c in combos if 0 < c["z_exit"] < c["z_entry"] < c["z_stop"]]


def evaluate(config: dict, z: pd.Series, prices: pd.DataFrame, beta: pd.DataFrame, band: float = 0.0) -> dict:
    """Corre una configuración en un tramo y devuelve operaciones, Sharpe y P&L neto (sin guardar el libro de órdenes)."""
    result = run_backtest(prices, beta, generate_positions(z, config), band=band, keep_fills=False)
    curve, trades = result.equity, result.trades
    return {"n_trades": len(trades), "sharpe": sharpe_ratio(curve["daily_return"]),
            "net_pnl": float(curve["equity"].iloc[-1] - INITIAL_CAPITAL),
            "commission": float(curve["commission"].sum()), "borrow_cost": float(curve["borrow_cost"].sum()),
            "win_rate": float((trades["net_pnl"] > 0).mean()) if len(trades) else np.nan}


def objective(row: dict, n_min: int) -> float:
    """Sharpe de train, o PENALTY si la configuración tiene menos de n_min operaciones."""
    return row["sharpe"] if row["n_trades"] >= n_min and np.isfinite(row["sharpe"]) else PENALTY


def run_grid(z_train: pd.Series, prices: pd.DataFrame, beta: pd.DataFrame, n_min: int = N_MIN_DEFAULT) -> pd.DataFrame:
    """Evalúa toda la rejilla SOLO en train; una fila por configuración con su objetivo penalizado."""
    rows = []
    for config in grid_configurations():
        rows.append({**config, **evaluate(config, z_train, prices, beta)})
    grid = pd.DataFrame(rows)
    grid["viable"] = grid["n_trades"] >= n_min
    grid["objective"] = [objective(r, n_min) for r in grid.to_dict("records")]
    grid["neighborhood_median"] = neighborhood_medians(grid)
    return grid


def grid_index(grid: pd.DataFrame) -> np.ndarray:
    """Posición de cada configuración en la rejilla (un índice entero por dimensión)."""
    return np.column_stack([grid[p].map({v: i for i, v in enumerate(GRID[p])}).to_numpy() for p in PARAMS])


def neighborhood_medians(grid: pd.DataFrame) -> np.ndarray:
    """Mediana del objetivo en el vecindario de cada configuración (a lo más un paso por dimensión, incluida ella)."""
    index = grid_index(grid)
    values = grid["objective"].to_numpy()
    return np.array([np.median(values[np.all(np.abs(index - index[row]) <= 1, axis=1)]) for row in range(len(grid))])


def sharpe_standard_error(sharpe_annual: float, n_obs: int) -> float:
    """Error estándar del Sharpe anualizado con rendimientos iid (Lo, 2002)."""
    daily = sharpe_annual / math.sqrt(ANNUALIZATION)
    return math.sqrt((1 + daily ** 2 / 2) / n_obs) * math.sqrt(ANNUALIZATION)


def neighbors_of(index: np.ndarray, row: int) -> np.ndarray:
    """Máscara del vecindario de una configuración (a lo más un paso por dimensión), sin incluirla."""
    return np.all(np.abs(index - index[row]) <= 1, axis=1) & (np.arange(len(index)) != row)


def choose(grid: pd.DataFrame, n_min: int, n_obs: int) -> dict:
    """Elige la configuración final con la regla de meseta y documenta orillas y vecindario."""
    viable = grid[grid["viable"]]
    if viable.empty:
        raise ValueError(f"ninguna configuración llega a {n_min} operaciones en train "
                         f"(máximo alcanzable: {int(grid['n_trades'].max())})")
    index = grid_index(grid)
    best = viable.loc[viable["objective"].idxmax()]
    best_near = neighbors_of(index, best.name)
    tolerance = sharpe_standard_error(best["sharpe"], n_obs)
    plateau = bool(np.median(grid.loc[best_near, "objective"]) >= best["objective"] - tolerance)
    chosen = best if plateau else viable.loc[viable["neighborhood_median"].idxmax()]
    near = neighbors_of(index, chosen.name)
    near_viable = near & grid["viable"].to_numpy()
    return {
        "params": {p: (int(chosen[p]) if p == "max_hold_days" else float(chosen[p])) for p in PARAMS},
        "best_raw": {**{p: float(best[p]) for p in PARAMS}, "sharpe": float(best["sharpe"]),
                     "n_trades": int(best["n_trades"]), "neighbors": int(best_near.sum()),
                     "neighbors_median_objective": float(np.median(grid.loc[best_near, "objective"])),
                     "neighbors_viable_share": float(grid.loc[best_near, "viable"].mean())},
        "plateau_tolerance_sharpe": tolerance,
        "best_raw_is_plateau": plateau,
        "rule": "meseta si la mediana del objetivo del vecindario >= Sharpe del mejor - 1 error estandar (Lo, 2002); "
                "si no, centro de la mejor meseta = configuracion viable con la mayor mediana de su vecindario",
        "chosen_sharpe_train": float(chosen["sharpe"]), "chosen_n_trades_train": int(chosen["n_trades"]),
        "chosen_neighborhood_median": float(chosen["neighborhood_median"]), "chosen_neighbors": int(near.sum()),
        "chosen_neighbors_viable_share": float(grid.loc[near, "viable"].mean()),
        "chosen_is_plateau": bool(np.median(grid.loc[near, "objective"])
                                  >= chosen["objective"] - sharpe_standard_error(chosen["sharpe"], n_obs)),
        "chosen_neighbors_sharpe_range": [float(grid.loc[near_viable, "sharpe"].min()),
                                          float(grid.loc[near_viable, "sharpe"].max())],
        "on_edge": {p: bool(chosen[p] in (GRID[p][0], GRID[p][-1])) for p in PARAMS},
    }


def choose_band(params: dict, z_train: pd.Series, prices: pd.DataFrame, beta: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Banda de no-rebalanceo elegida SOLO en train con los umbrales congelados: la de mayor Sharpe neto de train."""
    rows = [{"band": band, **evaluate(params, z_train, prices, beta, band=band)} for band in BAND_GRID]
    table = pd.DataFrame(rows)
    best = table.loc[table["sharpe"].idxmax()]
    finite = [b for b in BAND_GRID if np.isfinite(b)]
    return table, {"band": float(best["band"]), "sharpe_train": float(best["sharpe"]),
                   "on_edge": bool(best["band"] in (BAND_GRID[0], BAND_GRID[-1])),
                   "largest_finite_band": finite[-1]}
