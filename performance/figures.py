"""Figuras del bloque de Paula: 7 (curvas de capital) y 8 (distribución de retornos por operación).

Dueño: Paula
Lee: (se lo pasa performance.run) equity.csv y trades.csv
Escribe: artifacts/figures/fig7_curvas_capital.png, fig8_distribucion_retornos.png
Reglas: título, ejes etiquetados, leyenda legible y fronteras train/test/valid dibujadas y etiquetadas. Usa el estilo,
los colores y las fronteras de cointegration/figures.py (Milca) para que todas las figuras combinen.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
from scipy import stats

from cointegration.figures import INK, MUTED, PERIODS, SERIES_COLORS, STYLE, boundary_handle, mark_periods, save

PERIOD_COLORS = dict(zip(PERIODS, SERIES_COLORS[:3]))


def plot_equity_curves(equity: pd.DataFrame, splits: dict, capital: float):
    """Figura 7: equity de cada tramo (cada uno empieza con el capital inicial) y su drawdown debajo."""
    with plt.rc_context(STYLE):
        fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
        for period in PERIODS:
            part = equity[equity["period"] == period]
            change = part["equity"].iloc[-1] / capital - 1
            top.plot(part.index, part["equity"], color=PERIOD_COLORS[period], linewidth=1.1,
                     label=f"{period}: {part.index[0]:%Y-%m-%d} a {part.index[-1]:%Y-%m-%d}, rendimiento {change:+.2%}")
            bottom.fill_between(part.index, part["drawdown"], 0, color=PERIOD_COLORS[period], alpha=0.35, linewidth=0)
        top.axhline(capital, color=MUTED, linewidth=0.8, linestyle=":")
        top.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1e6:.2f} M"))
        bottom.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        for ax in (top, bottom):
            mark_periods(ax, equity.index, splits)
        top.set_title("Figura 7 · Curvas de capital neto de costos por tramo (train, test y valid; cada tramo empieza "
                      "con $1,000,000)", pad=16)
        top.set_ylabel("equity (USD)")
        bottom.set_ylabel("drawdown")
        bottom.set_xlabel("fecha")
        handles, _ = top.get_legend_handles_labels()
        handles += [Line2D([], [], color=MUTED, linestyle=":", label="capital inicial $1,000,000"), boundary_handle()]
        top.legend(handles=handles, loc="upper left", fontsize=8)
    return save(fig, "fig7_curvas_capital.png")


def plot_trade_returns(trades: pd.DataFrame, splits: dict):
    """Figura 8: histograma de retornos netos por operación por tramo con una normal superpuesta, y los retornos en el
    tiempo con las fronteras entre tramos."""
    returns = trades["return"] * 100
    edges = np.histogram_bin_edges(returns, bins=16)
    grid = np.linspace(edges[0], edges[-1], 300)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(13, 8))
        layout = fig.add_gridspec(2, 3, height_ratios=[1.2, 1])
        for col, period in enumerate(PERIODS):
            ax = fig.add_subplot(layout[0, col])
            values = returns[trades["period"] == period]
            ax.hist(values, bins=edges, color=PERIOD_COLORS[period], alpha=0.75, edgecolor="white", label="operaciones")
            if len(values) > 1:
                mean, std = values.mean(), values.std(ddof=1)
                ax.plot(grid, stats.norm.pdf(grid, mean, std) * len(values) * np.diff(edges)[0], color=INK,
                        linewidth=1.1, label=f"normal (μ = {mean:.2f}%, σ = {std:.2f}%)")
            ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
            ax.set_title(f"{period}: {len(values)} operaciones", pad=8)
            ax.set_xlabel("retorno neto por operación (% del equity de entrada)")
            ax.set_ylabel("número de operaciones")
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(fontsize=7, loc="upper left")
        ax = fig.add_subplot(layout[1, :])
        for period in PERIODS:
            part = trades[trades["period"] == period]
            ax.bar(part["exit_date"], part["return"] * 100, width=12, color=PERIOD_COLORS[period], label=period)
        ax.axhline(0, color=MUTED, linewidth=0.8)
        span = pd.DatetimeIndex(trades["entry_date"]).union(pd.DatetimeIndex(trades["exit_date"]))
        mark_periods(ax, span, splits)
        ax.set_title("Retorno neto de cada operación en la fecha de salida", pad=14)
        ax.set_xlabel("fecha de salida")
        ax.set_ylabel("retorno neto (%)")
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles + [boundary_handle()], loc="lower left", fontsize=8, ncol=4)
        fig.suptitle("Figura 8 · Distribución de retornos netos por operación (train, test y valid)", fontweight="bold")
        fig.tight_layout()
    return save(fig, "fig8_distribucion_retornos.png")
