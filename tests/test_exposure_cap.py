"""El tamaño de posición nunca supera el 80% del capital, en ningún día de ningún periodo (Paula).

Se verifica: (1) en los artefactos de los tres tramos, recalculando la exposición desde el libro de órdenes y los
precios; (2) en un escenario sintético adverso (la pata corta se dispara y el equity cae) con rebalanceo diario y sin
rebalanceo por banda, donde el tope tiene que forzar la reducción; (3) en la función de sizing con la comisión ya pagada.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.costs import COMMISSION_RATE
from backtest.engine import run_backtest
from backtest.walk_forward import load_inputs
from strategy.sizing import EXPOSURE_CAP, target_gross, target_shares, target_values

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "backtest"
TOL = 1e-12


def adverse_scenario() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Largo el spread mientras las cuatro patas cortas suben 90%: el equity cae y la exposición tiende a crecer."""
    dates = pd.bdate_range("2021-01-01", periods=120)
    tickers = ["a", "b", "c", "d", "e", "f"]
    prices = pd.DataFrame(50.0, index=dates, columns=tickers)
    prices[["c", "d", "e", "f"]] = np.repeat(50.0 * np.exp(np.linspace(0, 0.9, 120))[:, None], 4, axis=1)
    beta = pd.DataFrame([[0.3, 1.0, -0.5, -0.6, -0.2, -0.3]] * 120, index=dates, columns=tickers)
    positions = pd.DataFrame({"position": [0] + [1] * 118 + [0], "holding_days": [0] + list(range(1, 119)) + [0],
                              "exit_reason": [""] * 119 + ["senal"]}, index=dates)
    return prices, beta, positions


def test_position_size_never_exceeds_80_percent():
    """Sizing con la comisión ya pagada, escenario adverso con ambas políticas y los tres tramos guardados."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        equity, current = rng.uniform(1e5, 2e6), rng.uniform(0, 1.6e6)
        beta, prices = rng.normal(0, 1, 6), rng.uniform(5, 200, 6)
        gross_target = target_gross(equity, current, COMMISSION_RATE)
        shares = target_shares(target_values(int(rng.choice([-1, 1])), beta, gross_target), prices)
        gross = np.abs(shares * prices).sum()
        assert gross <= EXPOSURE_CAP * (equity - COMMISSION_RATE * (current + gross)) + TOL

    prices, beta, positions = adverse_scenario()
    for band in (0.0, np.inf):
        curve = run_backtest(prices, beta, positions, band=band).equity
        assert curve["equity"].iloc[-1] < 1_000_000
        assert (curve["exposure_ratio"] <= EXPOSURE_CAP + TOL).all(), f"band={band}"

    assert (ARTIFACTS / "equity.csv").exists(), "faltan los artefactos del backtest: corre python main.py"
    equity = pd.read_csv(ARTIFACTS / "equity.csv", index_col="date", parse_dates=True)
    fills = pd.read_csv(ARTIFACTS / "fills.csv", parse_dates=["date"])
    group_prices = load_inputs().prices
    assert set(equity["period"]) == {"train", "test", "valid"}
    assert (equity["exposure_ratio"] <= EXPOSURE_CAP + TOL).all()
    for period, part in equity.groupby("period"):
        flows = fills[fills["period"] == period].pivot_table(index="date", columns="ticker", values="shares",
                                                             aggfunc="sum")
        shares = flows.reindex(index=part.index, columns=group_prices.columns).fillna(0.0).cumsum()
        gross = (shares * group_prices.loc[part.index]).abs().sum(axis=1)
        assert (gross <= EXPOSURE_CAP * part["equity"] + 1e-6).all(), period
        assert (gross / part["equity"]).max() > 0.79, f"{period}: la prueba sería vacía si nunca se acerca al tope"
