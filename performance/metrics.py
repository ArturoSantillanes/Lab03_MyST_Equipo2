"""Sharpe, Sortino, Calmar, drawdown máximo, win rate y profit factor.

Dueño: Paula
Lee: Curva de capital y operaciones.
Escribe: Métricas (van a metrics.json).
Reglas: Reportar por periodo (train, test, valid).

Convenciones declaradas:
- Rendimientos diarios simples del equity: r_t = E_t/E_{t−1} − 1 (el primero contra el capital inicial).
- Tasa libre de riesgo = 0: el motor no paga intereses sobre el efectivo y la estrategia es long/short casi
  autofinanciada, así que el rendimiento en exceso es el propio rendimiento.
- Factor de anualización = 252 días hábiles (√252 para Sharpe y Sortino; 252/n para el CAGR).
- Sharpe = media/desviación estándar muestral (ddof = 1)·√252.
- Sortino = media/desviación a la baja·√252, con desviación a la baja = √(media de min(r_t, 0)²) sobre TODOS los días.
- Drawdown máximo = mínimo de E_t/máx_{s≤t}(E_s, capital inicial) − 1 (negativo).
- Calmar = CAGR / |drawdown máximo|.
- Win rate = fracción de operaciones con P&L neto > 0; profit factor = Σ ganancias netas / |Σ pérdidas netas|.
- Deflated Sharpe Ratio (Bailey y López de Prado, 2014) con el Sharpe por día (sin anualizar).
"""
import math

import numpy as np
import pandas as pd
from scipy import stats

ANNUALIZATION = 252
RISK_FREE_RATE = 0.0
EULER_GAMMA = 0.5772156649015329
ZERO_TOL = 1e-12  # desviaciones menores son ruido de punto flotante de una serie constante


def sharpe_ratio(returns: pd.Series, periods: int = ANNUALIZATION) -> float:
    """Sharpe anualizado con tasa libre de riesgo 0; NaN si la desviación es 0 o hay menos de 2 días."""
    r = pd.Series(returns).dropna()
    std = r.std(ddof=1)
    if len(r) < 2 or not np.isfinite(std) or std < ZERO_TOL:
        return float("nan")
    return float((r.mean() - RISK_FREE_RATE / periods) / std * math.sqrt(periods))


def sortino_ratio(returns: pd.Series, periods: int = ANNUALIZATION) -> float:
    """Sortino anualizado; NaN si no hay ningún día negativo."""
    r = pd.Series(returns).dropna()
    downside = math.sqrt((np.minimum(r, 0.0) ** 2).mean()) if len(r) else 0.0
    if downside < ZERO_TOL:
        return float("nan")
    return float((r.mean() - RISK_FREE_RATE / periods) / downside * math.sqrt(periods))


def drawdown(equity: pd.Series, initial_capital: float) -> pd.Series:
    """Drawdown diario contra el máximo previo (incluido el capital inicial)."""
    return equity / equity.cummax().clip(lower=initial_capital) - 1


def max_drawdown(equity: pd.Series, initial_capital: float) -> float:
    """Drawdown máximo (número negativo o 0)."""
    return float(drawdown(equity, initial_capital).min())


def cagr(equity: pd.Series, initial_capital: float, periods: int = ANNUALIZATION) -> float:
    """Rendimiento anual compuesto: (E_final/E_0)^(252/n) − 1, con n días del tramo."""
    return float((equity.iloc[-1] / initial_capital) ** (periods / len(equity)) - 1)


def calmar_ratio(equity: pd.Series, initial_capital: float) -> float:
    """CAGR / |drawdown máximo|; NaN si no hubo drawdown."""
    mdd = max_drawdown(equity, initial_capital)
    return float(cagr(equity, initial_capital) / abs(mdd)) if mdd < 0 else float("nan")


def win_rate(trade_pnl: pd.Series) -> float:
    """Fracción de operaciones con P&L neto positivo; NaN sin operaciones."""
    pnl = pd.Series(trade_pnl).dropna()
    return float((pnl > 0).mean()) if len(pnl) else float("nan")


def profit_factor(trade_pnl: pd.Series) -> float:
    """Ganancias netas / |pérdidas netas|; inf si no hubo pérdidas, NaN sin operaciones."""
    pnl = pd.Series(trade_pnl).dropna()
    if pnl.empty:
        return float("nan")
    losses = -pnl[pnl < 0].sum()
    return float(pnl[pnl > 0].sum() / losses) if losses > 0 else float("inf")


def expected_max_sharpe(n_trials: float, var_trials: float) -> float:
    """Sharpe máximo esperado por azar entre n pruebas independientes (umbral SR₀ del Deflated Sharpe Ratio)."""
    if n_trials <= 1:
        return 0.0
    return float(math.sqrt(var_trials) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                                          + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * math.e))))


def probabilistic_sharpe(sr: float, sr_benchmark: float, n_obs: int, skew: float, kurtosis: float) -> float:
    """P(Sharpe verdadero > benchmark) con corrección por asimetría y curtosis (Sharpe por día, curtosis no en exceso)."""
    denominator = 1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2
    return float(stats.norm.cdf((sr - sr_benchmark) * math.sqrt(n_obs - 1) / math.sqrt(denominator)))


def deflated_sharpe(returns: pd.Series, n_trials: float, var_trials: float) -> dict:
    """Deflated Sharpe Ratio: PSR contra el Sharpe máximo esperado por azar con n_trials pruebas."""
    r = pd.Series(returns).dropna()
    sr = float(r.mean() / r.std(ddof=1))
    skew, kurt = float(stats.skew(r)), float(stats.kurtosis(r, fisher=False))
    sr0 = expected_max_sharpe(n_trials, var_trials)
    return {"n_trials": float(n_trials), "sr_daily": sr, "sr0_daily": sr0, "sr0_annualized": sr0 * math.sqrt(ANNUALIZATION),
            "skew": skew, "kurtosis": kurt, "n_obs": len(r),
            "dsr": probabilistic_sharpe(sr, sr0, len(r), skew, kurt)}


def period_metrics(curve: pd.DataFrame, trades: pd.DataFrame, initial_capital: float) -> dict:
    """Todas las métricas de un tramo a partir de su curva diaria y sus operaciones."""
    equity, returns = curve["equity"], curve["daily_return"]
    invested = curve["position"] != 0
    return {
        "start": curve.index[0].strftime("%Y-%m-%d"), "end": curve.index[-1].strftime("%Y-%m-%d"),
        "n_days": len(curve), "final_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / initial_capital - 1), "cagr": cagr(equity, initial_capital),
        "volatility_annual": float(returns.std(ddof=1) * math.sqrt(ANNUALIZATION)),
        "sharpe": sharpe_ratio(returns), "sortino": sortino_ratio(returns),
        "calmar": calmar_ratio(equity, initial_capital), "max_drawdown": max_drawdown(equity, initial_capital),
        "n_trades": len(trades), "win_rate": win_rate(trades["net_pnl"]),
        "profit_factor": profit_factor(trades["net_pnl"]),
        "avg_trade_return": float(trades["return"].mean()) if len(trades) else float("nan"),
        "avg_holding_days": float(trades["holding_days"].mean()) if len(trades) else float("nan"),
        "share_days_invested": float(invested.mean()),
        "avg_exposure_all_days": float(curve["exposure_ratio"].mean()),
        "avg_exposure_when_invested": float(curve.loc[invested, "exposure_ratio"].mean()) if invested.any() else 0.0,
        "max_exposure": float(curve["exposure_ratio"].max()),
        "pnl_attribution": {
            "gross_pnl": float(curve["gross_pnl"].sum()), "commission": float(curve["commission"].sum()),
            "borrow_cost": float(curve["borrow_cost"].sum()),
            "net_pnl": float(curve["gross_pnl"].sum() - curve["commission"].sum() - curve["borrow_cost"].sum()),
        },
    }
