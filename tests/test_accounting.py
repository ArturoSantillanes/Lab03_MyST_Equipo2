"""Prueba obligatoria del enunciado: contabilidad del motor de backtest (Paula).

1. Valor del portafolio = efectivo + valor de todas las patas, verificado DÍA A DÍA. El efectivo y las acciones se
   reconstruyen de forma independiente a partir del libro de órdenes (fills), no se leen del motor.
2. Costo total registrado = Σ por pata de (comisión 0.125% × valor operado en cada operación) + devengo del préstamo
   sobre las patas cortas (0.25% anual, ACT/360), recalculado desde las acciones reconstruidas.
Se prueba sobre datos sintéticos (con cambios de signo de β y ambas políticas de rebalanceo), sobre train real y sobre
los artefactos de los tres tramos (artifacts/backtest/), sin volver a evaluar test ni valid.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.costs import BORROW_RATE_ANNUAL, COMMISSION_RATE, DAY_COUNT_BASIS
from backtest.engine import run_backtest
from backtest.walk_forward import load_inputs, period_z
from strategy import generate_positions
from strategy.sizing import INITIAL_CAPITAL

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "backtest"
TICKERS = ["a", "b", "c", "d", "e", "f"]


def synthetic(seed: int = 7, n: int = 400):
    """Precios en niveles, β̂_t que cambia de signo en una pata y posiciones con entradas y salidas en ambos sentidos."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(40 * np.exp(np.cumsum(rng.normal(0, 0.02, (n, 6)), axis=0)), index=dates, columns=TICKERS)
    base = np.array([0.3, 1.0, -0.5, -0.6, -0.2, -0.3])
    beta = pd.DataFrame(base + rng.normal(0, 0.05, (n, 6)), index=dates, columns=TICKERS)
    beta["e"] = np.linspace(-0.3, 0.3, n)
    z = pd.Series(2.5 * np.sin(np.arange(n) / 9.0) + rng.normal(0, 0.3, n), index=dates)
    positions = generate_positions(z, {"z_entry": 1.5, "z_exit": 0.3, "z_stop": 3.0, "max_hold_days": 20})
    return prices, beta, positions


def reconstruct(equity: pd.DataFrame, fills: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Acciones, efectivo y préstamo reconstruidos desde el libro de órdenes, día por día."""
    dates = equity.index
    flows = fills.pivot_table(index="date", columns="ticker", values="shares", aggfunc="sum")
    shares = flows.reindex(index=dates, columns=prices.columns, fill_value=0.0).fillna(0.0).cumsum()
    px = prices.loc[dates, shares.columns]
    short_value = (-(shares * px).clip(upper=0)).sum(axis=1)
    gaps = pd.Series(np.r_[0, np.diff(dates.values).astype("timedelta64[D]").astype(int)], index=dates)
    borrow = BORROW_RATE_ANNUAL * short_value.shift(fill_value=0.0) * gaps / DAY_COUNT_BASIS
    fees = (COMMISSION_RATE * fills["value"].abs()).groupby(fills["date"]).sum().reindex(dates, fill_value=0.0)
    trade_cash = fills.groupby("date")["value"].sum().reindex(dates, fill_value=0.0)
    cash = INITIAL_CAPITAL - (trade_cash + fees + borrow).cumsum()
    return pd.DataFrame({"cash": cash, "legs": (shares * px).sum(axis=1), "borrow": borrow, "fees": fees})


def check_identity(equity: pd.DataFrame, fills: pd.DataFrame, prices: pd.DataFrame) -> None:
    """Las dos afirmaciones de la prueba sobre una corrida (un tramo)."""
    rebuilt = reconstruct(equity, fills, prices)
    # 1. Día a día: valor del portafolio = efectivo + valor de las patas (reconstruidos de forma independiente).
    np.testing.assert_allclose(equity["equity"], rebuilt["cash"] + rebuilt["legs"], rtol=0, atol=1e-6)
    np.testing.assert_allclose(equity["cash"], rebuilt["cash"], rtol=0, atol=1e-6)
    # Y la ecuación de flujo: E_t = E_{t-1} + P&L bruto - comisión - préstamo.
    previous = equity["equity"].shift(fill_value=INITIAL_CAPITAL)
    np.testing.assert_allclose(equity["equity"], previous + equity["gross_pnl"] - equity["commission"]
                               - equity["borrow_cost"], rtol=0, atol=1e-6)
    # 2. Costos: número de operaciones por pata × comisión sobre su valor, más el devengo sobre las patas cortas.
    assert int(equity["n_leg_trades"].sum()) == len(fills)
    expected_commission = float((COMMISSION_RATE * fills["value"].abs()).sum())
    assert equity["commission"].sum() == pytest.approx(expected_commission, rel=1e-12, abs=1e-9)
    assert fills["commission"].sum() == pytest.approx(expected_commission, rel=1e-12, abs=1e-9)
    np.testing.assert_allclose(equity["borrow_cost"], rebuilt["borrow"], rtol=0, atol=1e-9)
    total = equity["commission"].sum() + equity["borrow_cost"].sum()
    assert total == pytest.approx(expected_commission + rebuilt["borrow"].sum(), rel=1e-12)


def test_portfolio_equals_cash_plus_legs_and_costs_match():
    """Identidad diaria y costos: sintético (rebalanceo diario y por banda), train real y los tres tramos guardados."""
    prices, beta, positions = synthetic()
    for band in (0.0, 0.05):
        result = run_backtest(prices, beta, positions, band=band)
        assert len(result.trades) >= 5 and set(result.trades["direction"]) == {"largo", "corto"}
        assert result.equity["borrow_cost"].sum() > 0
        check_identity(result.equity, result.fills, prices)
        # Las operaciones suman lo mismo que la curva: P&L neto de las operaciones = cambio del equity.
        assert result.trades["net_pnl"].sum() == pytest.approx(result.equity["equity"].iloc[-1] - INITIAL_CAPITAL,
                                                               abs=1e-6)

    inputs = load_inputs()
    train = run_backtest(inputs.prices, inputs.beta, generate_positions(period_z(inputs, "train"),
                                                                        inputs.params_default))
    check_identity(train.equity, train.fills, inputs.prices)

    assert (ARTIFACTS / "fills.csv").exists(), "faltan los artefactos del backtest: corre python main.py"
    equity = pd.read_csv(ARTIFACTS / "equity.csv", index_col="date", parse_dates=True)
    fills = pd.read_csv(ARTIFACTS / "fills.csv", parse_dates=["date"])
    equity["n_leg_trades"] = pd.read_csv(ARTIFACTS / "costs.csv", index_col="date", parse_dates=True)["n_leg_trades"]
    assert set(equity["period"]) == {"train", "test", "valid"}
    for period in ("train", "test", "valid"):
        check_identity(equity[equity["period"] == period], fills[fills["period"] == period], inputs.prices)
