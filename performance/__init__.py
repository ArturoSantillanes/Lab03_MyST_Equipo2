"""Etapa 'performance': calcula las métricas por periodo.

Dueño: Paula
Lee: artifacts/backtest/equity.csv, trades.csv
Escribe: artifacts/backtest/metrics.json, rolling_sharpe.csv
Reglas: Toda cifra dice si es train, test o valid.

También lee grid_search.csv, params.json, rebalance_comparison.csv y selected_group.json (conteo de pruebas) para el
Deflated Sharpe Ratio, y escribe las figuras 7 (curvas de capital) y 8 (distribución de retornos por operación).
No vuelve a correr el backtest.
"""
import math
from pathlib import Path

import pandas as pd

from backtest.io import read_json, write_json
from backtest.walk_forward import PERIODS
from performance.figures import plot_equity_curves, plot_trade_returns
from performance.metrics import ANNUALIZATION, RISK_FREE_RATE, deflated_sharpe, period_metrics
from performance.rolling import WINDOW, rolling_table

ROOT = Path(__file__).resolve().parent.parent
BACKTEST_DIR = ROOT / "artifacts" / "backtest"


def deflated_sharpe_report(equity: pd.DataFrame, grid: pd.DataFrame, multiple_testing: dict) -> dict:
    """DSR de train con distintos conteos de pruebas; test y valid se reportan como PSR (una sola prueba cada uno).

    La varianza de los Sharpe entre pruebas sale de las configuraciones viables de la rejilla (por día, sin anualizar).
    """
    viable = grid[grid["viable"]]
    var_trials = float((viable["sharpe"] / math.sqrt(ANNUALIZATION)).var(ddof=1))
    n_grid = len(grid)
    n_johansen = multiple_testing["johansen_tests_train"]
    train = equity.loc[equity["period"] == "train", "daily_return"]
    counts = {"solo_rejilla": n_grid, "solo_johansen": n_johansen, "johansen_mas_rejilla": n_johansen + n_grid,
              "johansen_por_rejilla": n_johansen * n_grid}
    return {
        "method": "Deflated Sharpe Ratio (Bailey y López de Prado, 2014); Sharpe por día; varianza entre pruebas de "
                  "las configuraciones viables de la rejilla de train",
        "var_trials_daily_sharpe": var_trials, "n_grid_configurations": n_grid, "n_grid_viable": int(len(viable)),
        "johansen_tests_train": n_johansen,
        "expected_false_positives_johansen_5pct": multiple_testing["expected_false_positives_johansen_5pct"],
        "train": {name: deflated_sharpe(train, n, var_trials) for name, n in counts.items()},
        "out_of_sample_psr": {p: deflated_sharpe(equity.loc[equity["period"] == p, "daily_return"], 1, var_trials)
                              for p in ("test", "valid")},
        "main": "train con johansen_por_rejilla (cota conservadora: cada configuración pudo combinarse con cada grupo)",
    }


def run() -> None:
    """Métricas por tramo, atribución de P&L, exposición, Sharpe móvil, DSR y figuras 7 y 8."""
    equity = pd.read_csv(BACKTEST_DIR / "equity.csv", index_col="date", parse_dates=True)
    trades = pd.read_csv(BACKTEST_DIR / "trades.csv", parse_dates=["entry_date", "exit_date"])
    grid = pd.read_csv(BACKTEST_DIR / "grid_search.csv")
    params = read_json(BACKTEST_DIR / "params.json")
    comparison = pd.read_csv(BACKTEST_DIR / "rebalance_comparison.csv")
    selection = read_json(ROOT / "artifacts" / "cointegration" / "selected_group.json")
    splits = read_json(ROOT / "data" / "processed" / "splits.json")
    capital = params["assumptions"]["initial_capital"]

    by_period = {p: period_metrics(equity[equity["period"] == p], trades[trades["period"] == p], capital)
                 for p in PERIODS}
    for p in PERIODS:
        by_period[p]["exit_reasons"] = trades.loc[trades["period"] == p, "exit_reason"].value_counts().to_dict()
    metrics = {
        "conventions": {"risk_free_rate": RISK_FREE_RATE, "annualization_days": ANNUALIZATION,
                        "returns": "simples diarios del equity; el primero contra el capital inicial",
                        "rolling_window_days": WINDOW,
                        "periods": "cada tramo es una corrida independiente que empieza con el capital inicial; "
                                   "train empieza el primer día operable (2015-05-21)"},
        "params": {k: params[k] for k in ("z_entry", "z_exit", "z_stop", "max_hold_days", "policy", "band")},
        "by_period": by_period,
        "rebalance_comparison": comparison.to_dict(orient="records"),
        "deflated_sharpe": deflated_sharpe_report(equity, grid, selection["multiple_testing"]),
        "validation_evaluations": params["validation_evaluations"],
    }
    write_json(BACKTEST_DIR / "metrics.json", metrics)
    rolling_table(equity).to_csv(BACKTEST_DIR / "rolling_sharpe.csv", date_format="%Y-%m-%d", index_label="date")
    plot_equity_curves(equity, splits, capital)
    plot_trade_returns(trades, splits)
    for p in PERIODS:
        m = by_period[p]
        print(f"  {p}: Sharpe {m['sharpe']:.2f}, Sortino {m['sortino']:.2f}, MDD {m['max_drawdown']:.2%}, "
              f"{m['n_trades']} operaciones, P&L neto {m['pnl_attribution']['net_pnl']:,.0f}")
