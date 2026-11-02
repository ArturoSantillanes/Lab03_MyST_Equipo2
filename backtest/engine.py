"""Motor de backtest multi-pata: efectivo, patas, comisiones y devengo de préstamo.

Dueño: Paula
Lee: Precios, señales y pesos.
Escribe: Curva de capital y operaciones.
Reglas: Valor del portafolio = efectivo + valor de las patas. Sin look-ahead.

Supuesto de ejecución (declarado): la señal d_t y los pesos β̂_t se conocen al cierre del día t y se ejecutan a
ese mismo precio de cierre P_t (precio ajustado en niveles). Las acciones resultantes se mantienen del cierre de t al
cierre de t+1, así que d_t gana o pierde el cambio de precio de t a t+1, nunca el de t: en el día t+1 el P&L bruto es
Σ_i acciones_{t,i}·(P_{t+1,i} − P_{t,i}), calculado con las acciones del día anterior ANTES de operar.

Orden de cada día t (al cierre):
  1. P&L bruto con las acciones de t−1 y devengo del préstamo sobre las patas cortas de t−1 (ACT/360).
  2. Equity antes de operar E_t = efectivo + Σ acciones·P_t (capital disponible del día para el sizing).
  3. Acciones objetivo según d_t, β̂_t y E_t; se opera la diferencia por pata y se paga la comisión de cada pata.
Contabilidad en efectivo: comprar resta efectivo, vender en corto lo suma; el valor del portafolio es siempre
efectivo + Σ acciones_i·P_i (las cortas con acciones negativas). El efectivo no devenga intereses (tasa 0).

Política de rebalanceo: band = 0 rebalancea cada día a los pesos β̂_t del día; band > 0 solo rebalancea si alguna pata
se aleja de su objetivo más que band (strategy.sizing.needs_rebalance) o si la exposición bruta pasa del tope de 80%.
El último día del tramo se cierra cualquier posición abierta (motivo 'fin_de_periodo') para que cada tramo se evalúe
por separado, sin posiciones que crucen de train a test o de test a valid.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backtest.costs import COMMISSION_RATE, borrow_accrual, commission
from performance.metrics import drawdown
from strategy.sizing import (EXPOSURE_CAP, INITIAL_CAPITAL, gross_exposure, needs_rebalance, target_gross,
                             target_shares, target_values)

END_OF_PERIOD = "fin_de_periodo"
DIRECTION_LABEL = {1: "largo", -1: "corto"}


@dataclass
class BacktestResult:
    """Salidas del motor: curva diaria, libro de órdenes por pata y una fila por operación."""
    equity: pd.DataFrame
    fills: pd.DataFrame
    trades: pd.DataFrame


def close_at_end(positions: pd.DataFrame) -> pd.DataFrame:
    """Copia de las posiciones con la posición del último día forzada a 0 (cierre por fin de tramo)."""
    out = positions[["position", "holding_days", "exit_reason"]].copy()
    out["exit_reason"] = out["exit_reason"].fillna("").astype(str)
    last = out.index[-1]
    previous = int(out["position"].iloc[-2]) if len(out) > 1 else 0
    if int(out.at[last, "position"]) != 0:
        out.at[last, "position"] = 0
        out.at[last, "exit_reason"] = END_OF_PERIOD if previous != 0 else ""
        out.at[last, "holding_days"] = 0
    return out


def run_backtest(prices: pd.DataFrame, beta: pd.DataFrame, positions: pd.DataFrame, band: float = 0.0,
                 initial_capital: float = INITIAL_CAPITAL, keep_fills: bool = True) -> BacktestResult:
    """Corre el motor día a día sobre las fechas de `positions` (un tramo).

    prices: precios ajustados en niveles; beta: β̂_t por activo (mismas columnas); positions: salida de
    strategy.generate_positions. band: banda de no-rebalanceo (0 = diario).
    """
    positions = close_at_end(positions)
    dates = positions.index
    tickers = list(beta.columns)
    price = prices.loc[dates, tickers].to_numpy(dtype=float)
    weights = beta.loc[dates, tickers].to_numpy(dtype=float)
    direction = positions["position"].to_numpy(dtype=int)
    reasons = positions["exit_reason"].to_numpy()
    held_days = positions["holding_days"].to_numpy(dtype=int)
    gaps = np.r_[0, np.diff(dates.values).astype("timedelta64[D]").astype(int)]

    n, k = price.shape
    shares = np.zeros(k)
    cash = float(initial_capital)
    daily = np.zeros((n, 11))
    fills, trades = [], []
    trade = None
    for t in range(n):
        p = price[t]
        gross_pnl = float(shares @ (p - price[t - 1])) if t > 0 else 0.0
        short_prev = float(-(shares[shares < 0] * price[t - 1][shares < 0]).sum()) if t > 0 else 0.0
        borrow = borrow_accrual(short_prev, gaps[t]) if short_prev > 0 else 0.0
        cash -= borrow
        equity_pre = cash + float(shares @ p)
        if trade is not None:
            trade["gross_pnl"] += gross_pnl
            trade["borrow_cost"] += borrow

        d = direction[t]
        previous_d = direction[t - 1] if t > 0 else 0
        current_values = shares * p
        if d == 0:
            new_shares = np.zeros(k)
            kind = "salida"
        else:
            gross_target = target_gross(equity_pre, gross_exposure(shares, p), COMMISSION_RATE)
            desired = target_values(d, weights[t], gross_target)
            kind = "entrada" if previous_d == 0 else "rebalanceo"
            over_cap = gross_exposure(shares, p) > EXPOSURE_CAP * equity_pre
            if kind == "entrada" or band <= 0 or over_cap or needs_rebalance(current_values, desired, band):
                new_shares = target_shares(desired, p)
            else:
                new_shares = shares.copy()

        delta = new_shares - shares
        traded = np.abs(delta * p)
        fees = commission(delta * p)
        traded_legs = delta != 0
        cash -= float(delta @ p) + float(fees.sum())
        if kind == "entrada" and traded_legs.any():
            trade = {"entry_date": dates[t], "direction": DIRECTION_LABEL[d], "entry_equity": equity_pre,
                     "gross_pnl": 0.0, "commission": 0.0, "borrow_cost": 0.0, "n_rebalances": 0}
        if trade is not None:
            trade["commission"] += float(fees.sum())
            trade["n_rebalances"] += int(kind == "rebalanceo" and traded_legs.any())
        if keep_fills:
            for i in np.flatnonzero(traded_legs):
                fills.append((dates[t], tickers[i], kind, delta[i], p[i], delta[i] * p[i], fees[i]))
        shares = new_shares
        if trade is not None and kind == "salida" and previous_d != 0:
            trade.update(exit_date=dates[t], holding_days=int(held_days[t - 1]) + 1, exit_reason=reasons[t] or END_OF_PERIOD)
            trade["net_pnl"] = trade["gross_pnl"] - trade["commission"] - trade["borrow_cost"]
            trade["return"] = trade["net_pnl"] / trade["entry_equity"]
            trades.append(trade)
            trade = None

        values = shares * p
        long_value = float(values[values > 0].sum())
        short_value = float(-values[values < 0].sum())
        equity = cash + float(values.sum())
        daily[t] = (equity, cash, long_value, short_value, long_value + short_value, gross_pnl, float(fees.sum()),
                    borrow, int(traded_legs.sum()), float(traded.sum()), d)

    columns = ["equity", "cash", "long_value", "short_value", "gross_exposure", "gross_pnl", "commission",
               "borrow_cost", "n_leg_trades", "traded_value", "position"]
    curve = pd.DataFrame(daily, index=dates, columns=columns)
    curve[["n_leg_trades", "position"]] = curve[["n_leg_trades", "position"]].astype(int)
    curve["exposure_ratio"] = curve["gross_exposure"] / curve["equity"]
    curve["daily_return"] = curve["equity"].pct_change()
    curve.iloc[0, curve.columns.get_loc("daily_return")] = curve["equity"].iloc[0] / initial_capital - 1
    curve["drawdown"] = drawdown(curve["equity"], initial_capital)
    fills_frame = pd.DataFrame(fills, columns=["date", "ticker", "kind", "shares", "price", "value", "commission"])
    trade_columns = ["entry_date", "exit_date", "direction", "gross_pnl", "commission", "borrow_cost", "net_pnl",
                     "return", "holding_days", "exit_reason", "entry_equity", "n_rebalances"]
    trades_frame = pd.DataFrame(trades, columns=trade_columns)
    return BacktestResult(curve, fills_frame, trades_frame)
