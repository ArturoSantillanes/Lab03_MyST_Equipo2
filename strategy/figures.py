"""Figuras del bloque de Arturo: 2 (spread con señales), 5 (ganancia de Kalman de ambos filtros) y 6 (z-score).

Dueño: Arturo
Lee: (se lo pasa strategy.run) spread_t.csv, z_t.csv, gain_1.csv, gain_2.csv, posiciones y parámetros por defecto
Escribe: artifacts/figures/fig2_spread_senales.png, fig5_ganancia_kalman.png, fig6_zscore.png
Reglas: título, ejes etiquetados, leyenda legible y fronteras train/test/valid dibujadas y etiquetadas. La ganancia se
grafica en valor absoluto y escala LINEAL (la logarítmica no muestra negativos y haría mentir a la figura).
Usa el estilo, los colores y las fronteras de cointegration/figures.py (Milca) para que todas las figuras combinen.
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from cointegration.figures import (INK, MUTED, SERIES_COLORS, SHADE, STYLE, boundary_handle, mark_periods, save,
                                   shown_periods, ticker_colors)

LONG_COLOR, SHORT_COLOR = SERIES_COLORS[2], SERIES_COLORS[1]
EXIT_MARKERS = {"senal": ("o", "salida por señal (|z| < z_exit)"), "stop": ("X", "salida por stop (|z| > z_stop)"),
                "tenencia_maxima": ("s", "salida por tenencia máxima")}


def shade_burn_in(ax, burn_in: pd.Series) -> None:
    """Sombrea los días sin señal (burn-in de los filtros)."""
    dates = burn_in.index[burn_in.to_numpy()]
    if len(dates):
        ax.axvspan(dates[0], dates[-1], color=SHADE, alpha=0.6, linewidth=0)


def periods_label(index: pd.DatetimeIndex, splits: dict) -> str:
    """Tramos que aparecen en la figura, para el título."""
    return ", ".join(shown_periods(index, splits))


def plot_spread_signals(z_t: pd.DataFrame, positions: pd.DataFrame, params: dict, splits: dict, start: pd.Timestamp):
    """Figura 2: spread S_t con μ̂_{t|t−1} ± z_entry·σ̂_{t|t−1}, entradas y salidas marcadas."""
    frame, pos = z_t.loc[start:], positions.loc[start:]
    previous = pos["position"].shift(fill_value=0)
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 5.2))
        shade_burn_in(ax, frame["burn_in"])
        ax.plot(frame.index, frame["spread"], color=SERIES_COLORS[0], linewidth=0.8, label="spread S_t (filtro 1)")
        ax.plot(frame.index, frame["mu_prev"], color=INK, linewidth=0.9, label="media previa μ̂_t|t−1 (filtro 2)")
        band = params["z_entry"] * frame["sigma_prev"]
        ax.fill_between(frame.index, frame["mu_prev"] - band, frame["mu_prev"] + band, color=SERIES_COLORS[0],
                        alpha=0.10, linewidth=0, label=f"μ̂ ± z_entry·σ̂ (z_entry = {params['z_entry']})")
        long_in = pos.index[(previous == 0) & (pos["position"] == 1)]
        short_in = pos.index[(previous == 0) & (pos["position"] == -1)]
        ax.scatter(long_in, frame.loc[long_in, "spread"], marker="^", s=55, color=LONG_COLOR, zorder=4,
                   label="entrada larga en el spread (z < −z_entry)")
        ax.scatter(short_in, frame.loc[short_in, "spread"], marker="v", s=55, color=SHORT_COLOR, zorder=4,
                   label="entrada corta en el spread (z > +z_entry)")
        for reason, (marker, label) in EXIT_MARKERS.items():
            days = pos.index[pos["exit_reason"] == reason]
            ax.scatter(days, frame.loc[days, "spread"], marker=marker, s=34, facecolor="none" if marker == "o" else INK,
                       edgecolor=INK, linewidth=0.9, zorder=4, label=label)
        mark_periods(ax, frame.index, splits)
        ax.set_title(f"Figura 2 · Spread del portafolio con señales de entrada y salida ({periods_label(frame.index, splits)})",
                     pad=16)
        ax.set_xlabel("fecha")
        ax.set_ylabel("spread S_t = β̂_tᵀ(log P_t − media de train)")
        handles, _ = ax.get_legend_handles_labels()
        handles += [Patch(color=SHADE, alpha=0.6, label="burn-in del filtro 2 (sin señales)"), boundary_handle()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)
    return save(fig, "fig2_spread_senales.png")


def gain_panel(ax, gain: pd.Series, burn_in: pd.Series, color: str, title: str, splits: dict) -> None:
    """|K_t| en escala lineal; el eje y se ajusta al tramo sin burn-in y se anota el máximo del burn-in."""
    values = gain.abs()
    ax.plot(values.index, values, color=color, linewidth=0.7)
    settled = values[~burn_in]
    ax.set_ylim(0, 1.15 * settled.max())
    ax.text(0.98, 0.97, f"máx. en burn-in: {values[burn_in].max():.2g}", transform=ax.transAxes, ha="right", va="top",
            fontsize=7, color=MUTED)
    shade_burn_in(ax, burn_in)
    mark_periods(ax, values.index, splits)
    ax.set_title(title, pad=14)
    ax.set_xlabel("fecha")
    ax.set_ylabel("|K_t| (escala lineal)")


def plot_kalman_gains(gain_1: pd.DataFrame, gain_2: pd.DataFrame, tickers: list[str], splits: dict):
    """Figura 5: |K_t| de cada componente del filtro 1 (pesos β) y del filtro 2 (μ y σ²)."""
    assets = [c for c in gain_1.columns if c not in ("period", "burn_in")]
    colors = ticker_colors(tickers)
    filter2 = gain_2.dropna(subset=["mu"])
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 8.5))
        grid = fig.add_gridspec(2, 2 * len(assets))
        for i, asset in enumerate(assets):
            ax = fig.add_subplot(grid[0, 2 * i:2 * i + 2])
            gain_panel(ax, gain_1[asset], gain_1["burn_in"], colors[asset], f"Filtro 1 · peso β de {asset.upper()}", splits)
        half = len(assets)
        for j, (column, title) in enumerate((("mu", "Filtro 2 · media latente μ_t"),
                                             ("sigma2", "Filtro 2 · varianza latente σ_t²"))):
            ax = fig.add_subplot(grid[1, j * half:(j + 1) * half])
            gain_panel(ax, filter2[column], filter2["burn_in"], INK if j == 0 else MUTED, title, splits)
        fig.suptitle(f"Figura 5 · Ganancia de Kalman |K_t| de ambos filtros ({periods_label(gain_1.index, splits)}): "
                     "qué tanto mueve cada observación la estimación", fontweight="bold")
        fig.legend(handles=[Line2D([], [], color=MUTED, linewidth=0.7, label="|K_t| (valor absoluto, escala lineal)"),
                            Patch(color=SHADE, alpha=0.6, label="burn-in de cada filtro"), boundary_handle()],
                   loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return save(fig, "fig5_ganancia_kalman.png")


def plot_zscore(z_t: pd.DataFrame, params: dict, splits: dict, start: pd.Timestamp):
    """Figura 6: z-score con las líneas de entrada, salida y stop."""
    frame = z_t.loc[start:]
    lines = (("z_entry", SERIES_COLORS[1], "--", "umbral de entrada ±z_entry"),
             ("z_exit", SERIES_COLORS[2], ":", "umbral de salida ±z_exit"),
             ("z_stop", INK, "-.", "stop ±z_stop"))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(13, 4.8))
        shade_burn_in(ax, frame["burn_in"])
        ax.plot(frame.index, frame["z"], color=SERIES_COLORS[0], linewidth=0.7, label="z_t (estadísticos previos del filtro 2)")
        handles = [Line2D([], [], color=SERIES_COLORS[0], linewidth=0.7, label="z_t (estadísticos previos del filtro 2)")]
        for key, color, style, label in lines:
            for sign in (1, -1):
                ax.axhline(sign * params[key], color=color, linestyle=style, linewidth=1.0)
            handles.append(Line2D([], [], color=color, linestyle=style, linewidth=1.0, label=f"{label} = {params[key]}"))
        mark_periods(ax, frame.index, splits)
        ax.set_title(f"Figura 6 · Z-score del spread con umbrales ({periods_label(frame.index, splits)})", pad=16)
        ax.set_xlabel("fecha")
        ax.set_ylabel("z_t = (S_t − μ̂_t|t−1) / σ̂_t|t−1")
        handles += [Patch(color=SHADE, alpha=0.6, label="burn-in del filtro 2 (sin señales)"), boundary_handle()]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8)
    return save(fig, "fig6_zscore.png")
