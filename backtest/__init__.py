"""Etapa 'backtest': corre el motor con costos sobre train, test y valid.

Dueño: Paula
Lee: data/processed/prices.csv, artifacts/kalman/*.csv, artifacts/strategy/params_default.json
Escribe: artifacts/backtest/equity.csv, trades.csv, costs.csv, params.json, grid_search.csv
Reglas: Capital inicial $1,000,000; comisión 0.125% por pata en entrada y salida; préstamo 0.25% anual con devengo diario. Umbrales solo con train.

Además escribe: fills.csv (libro de órdenes por pata), positions.csv (señales con los parámetros finales),
band_grid.csv (banda de no-rebalanceo, train), rebalance_comparison.csv (diario contra banda, mismas señales y tramos)
y exit_analysis.json (motivos de salida y verificación del stop sobre |z|).
Orden: rejilla en train → regla de meseta → banda en train → parámetros congelados → train → test → valid (una vez).
"""
from pathlib import Path

import pandas as pd

from backtest.calibration import (BAND_GRID, GRID, N_MIN_DEFAULT, choose, choose_band, evaluate, grid_configurations,
                                  run_grid)
from backtest.io import write_json
from backtest.costs import BORROW_RATE_ANNUAL, COMMISSION_RATE, DAY_COUNT_BASIS
from backtest.walk_forward import ValidationGuard, load_inputs, period_z, stack, stop_diagnostics, walk_forward
from performance.metrics import sharpe_ratio
from strategy.sizing import EXPOSURE_CAP, INITIAL_CAPITAL

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "backtest"
N_MIN = N_MIN_DEFAULT
STRATEGY_KEYS = ("z_entry", "z_exit", "z_stop", "max_hold_days")


def write_csv(frame: pd.DataFrame, name: str, index_label: str | None = "date") -> None:
    """CSV con fechas ISO y precisión completa (la prueba de contabilidad lo exige); sin índice si index_label es None."""
    frame.to_csv(OUT_DIR / name, date_format="%Y-%m-%d", index=index_label is not None, index_label=index_label)


def policy_summary(results: dict, policy: str, band: float) -> list[dict]:
    """Costos y desempeño de una política por tramo (para comparar diario contra banda)."""
    rows = []
    for period, (_, result) in results.items():
        curve = result.equity
        rows.append({"period": period, "policy": policy, "band": band, "n_trades": len(result.trades),
                     "gross_pnl": curve["gross_pnl"].sum(), "commission": curve["commission"].sum(),
                     "borrow_cost": curve["borrow_cost"].sum(),
                     "net_pnl": curve["equity"].iloc[-1] - INITIAL_CAPITAL,
                     "n_leg_trades": int(curve["n_leg_trades"].sum()), "traded_value": curve["traded_value"].sum(),
                     "n_rebalance_days": int(result.trades["n_rebalances"].sum()),
                     "sharpe": sharpe_ratio(curve["daily_return"])})
    return rows


def run() -> None:
    """Calibra en train, congela, evalúa train → test → valid y escribe los artefactos de artifacts/backtest/."""
    inputs = load_inputs()
    z_train = period_z(inputs, "train")
    grid = run_grid(z_train, inputs.prices, inputs.beta, N_MIN)
    choice = choose(grid, N_MIN, len(z_train))
    params = choice["params"]
    band_table, band_choice = choose_band(params, z_train, inputs.prices, inputs.beta)
    band = band_choice["band"]
    final_policy = "banda" if band > 0 else "diaria"
    policies = {"diaria": 0.0, "banda": band} if band > 0 else {"diaria": 0.0}

    guard = ValidationGuard()
    results = walk_forward(inputs, params, policies, guard)
    final = results[final_policy]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    equity = stack(final, "equity")
    equity_columns = ["period", "equity", "cash", "long_value", "short_value", "gross_exposure", "exposure_ratio",
                      "position", "daily_return", "drawdown", "gross_pnl", "commission", "borrow_cost"]
    write_csv(equity[equity_columns], "equity.csv")
    costs = equity[["period", "commission", "borrow_cost", "n_leg_trades", "traded_value", "short_value"]].copy()
    costs.insert(3, "total_cost", costs["commission"] + costs["borrow_cost"])
    write_csv(costs, "costs.csv")
    trades = stack(final, "trades").reset_index(drop=True)
    trades = trades[["entry_date", "exit_date", "period"] + [c for c in trades.columns
                                                            if c not in ("entry_date", "exit_date", "period")]]
    write_csv(trades, "trades.csv", index_label=None)
    fills = stack(final, "fills").reset_index(drop=True)
    write_csv(fills[["date", "period"] + [c for c in fills.columns if c not in ("date", "period")]], "fills.csv",
              index_label=None)
    positions = pd.concat([pos.assign(period=p, z=inputs.z.loc[pos.index]) for p, (pos, _) in final.items()])
    write_csv(positions[["period", "z", "position", "holding_days", "exit_reason"]], "positions.csv")
    write_csv(grid.assign(period="train"), "grid_search.csv", index_label=None)
    write_csv(band_table.assign(period="train"), "band_grid.csv", index_label=None)
    comparison = sum((policy_summary(results[name], name, b) for name, b in policies.items()), [])
    write_csv(pd.DataFrame(comparison), "rebalance_comparison.csv", index_label=None)
    write_json(OUT_DIR / "exit_analysis.json", stop_diagnostics(inputs, final, params))

    default_train = evaluate(inputs.params_default, z_train, inputs.prices, inputs.beta)
    train_trades = final["train"][1].trades
    write_json(OUT_DIR / "params.json", {
        **params, "period_used": "train",
        "policy": final_policy, "band": band,
        "n_min_trades": N_MIN, "n_min_lowered": False,
        "grid": {"values": GRID, "n_configurations": len(grid_configurations()), "n_viable": int(grid["viable"].sum()),
                 "objective": "Sharpe anualizado de rendimientos diarios netos en train (rebalanceo diario); "
                              "-1e9 si hay menos de n_min_trades operaciones"},
        "selection": choice,
        "max_hold_binding_train": {"exits_by_max_hold": int((train_trades["exit_reason"] == "tenencia_maxima").sum()),
                                   "longest_trade_days": int(train_trades["holding_days"].max())},
        "band_selection": {**band_choice, "grid": BAND_GRID,
                           "criterion": "mayor Sharpe neto de train con los umbrales ya congelados"},
        "defaults": {"params_default": {k: inputs.params_default[k] for k in STRATEGY_KEYS},
                     "train_with_default_params": default_train,
                     "fallbacks_to_default": 0,
                     "fallbacks_note": "el código no tiene ruta de respaldo: si ninguna configuración llega a "
                                       "n_min_trades, calibration.choose truena en vez de usar valores por defecto",
                     "chosen_equal_to_default": {k: bool(params[k] == inputs.params_default[k]) for k in STRATEGY_KEYS}},
        "validation_evaluations": guard.count, "validation_log": guard.log,
        "assumptions": {
            "execution": "señal y pesos al cierre de t, ejecución al cierre de t; d_t gana el cambio de t a t+1",
            "available_capital": "equity del día (marcado al cierre de t antes de operar)",
            "exposure_cap": EXPOSURE_CAP, "initial_capital": INITIAL_CAPITAL,
            "commission_rate_per_leg_per_trade": COMMISSION_RATE, "borrow_rate_annual": BORROW_RATE_ANNUAL,
            "borrow_day_count": f"ACT/{DAY_COUNT_BASIS} (días naturales desde el cierre anterior)",
            "shares": "enteras, truncadas hacia cero", "cash_interest": 0.0,
            "walk_forward": "ancla fija: calibrar en train, congelar y evaluar train -> test -> valid, cada tramo "
                            "independiente desde $1,000,000 y cerrado al final del tramo",
        },
    })
    print(f"  rejilla (train): {len(grid)} configuraciones, {int(grid['viable'].sum())} con >= {N_MIN} operaciones; "
          f"elegidos {params} (meseta del mejor: {choice['best_raw_is_plateau']}; orilla: {choice['on_edge']})")
    print(f"  banda (train): {band}; política final: {final_policy}; valid evaluada {guard.count} vez")
    for period, (_, result) in final.items():
        print(f"  {period}: {len(result.trades)} operaciones, equity final {result.equity['equity'].iloc[-1]:,.0f}")
