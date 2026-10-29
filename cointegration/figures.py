"""Figuras del bloque de Milca: 1 (log-precios), 3 (pesos β̂_t), 4 (eigenvector móvil) y diagnóstico del filtro 1.

Dueño: Milca
Lee: datos y artefactos de data/, cointegration/ y kalman/filter1_weights.py (se los pasan las etapas).
Escribe: artifacts/figures/fig1_log_precios.png, fig3_beta_t.png, fig4_eigenvector.png, fig_diagnostico_filtro1.png
Reglas: Título, ejes etiquetados, leyenda legible y fronteras train/test/valid dibujadas y etiquetadas.

Un panel por activo (small multiples): seis líneas encimadas no se leen y las escalas son distintas.
Cada activo conserva su color en todas las figuras. La ganancia de Kalman se grafica en valor absoluto y escala
LINEAL: la escala logarítmica no puede mostrar valores negativos y haría que la figura mintiera.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

FIGURES_DIR = Path(__file__).resolve().parent.parent / "artifacts" / "figures"
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, SHADE = "#0b0b0b", "#52514e", "#d9d8d3"
PERIODS = ("train", "test", "valid")
STYLE = {
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9,
    "legend.frameon": False,
}


def ticker_colors(tickers: list[str]) -> dict[str, str]:
    """Color fijo por activo, en el orden del grupo elegido."""
    return dict(zip(tickers, SERIES_COLORS))


def shown_periods(index: pd.DatetimeIndex, splits: dict) -> list[str]:
    """Tramos que aparecen en las fechas graficadas."""
    return [p for p in PERIODS if pd.Timestamp(splits[p]["start"]) <= index[-1]]


def mark_periods(ax, index: pd.DatetimeIndex, splits: dict) -> None:
    """Dibuja las fronteras entre tramos y escribe el nombre de cada tramo arriba del panel."""
    periods = shown_periods(index, splits)
    for name in periods[1:]:
        ax.axvline(pd.Timestamp(splits[name]["start"]), color=MUTED, linestyle="--", linewidth=0.8)
    for name in periods:
        start = max(pd.Timestamp(splits[name]["start"]), index[0])
        end = min(pd.Timestamp(splits[name]["end"]), index[-1])
        ax.text(start + (end - start) / 2, 1.0, name, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                color=MUTED, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def boundary_handle() -> Line2D:
    """Entrada de leyenda de las fronteras entre tramos."""
    return Line2D([], [], color=MUTED, linestyle="--", linewidth=0.8, label="frontera entre tramos (train | test | valid)")


def save(fig, name: str) -> Path:
    """Guarda en artifacts/figures/ y cierra la figura."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def asset_panels(frame: pd.DataFrame, splits: dict, title: str, ylabel: str, filename: str,
                 reference: dict | None = None, extra_handles: tuple = ()) -> Path:
    """Un panel por activo con su serie, las fronteras de tramos y, opcionalmente, una línea de referencia."""
    tickers = [c for c in frame.columns if c not in ("period", "burn_in", "similarity", "rank_trace")]
    colors = ticker_colors(tickers)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
        for ax, ticker in zip(axes.flat, tickers):
            ax.plot(frame.index, frame[ticker], color=colors[ticker], linewidth=1.0)
            if reference is not None:
                ax.axhline(reference[ticker], color=colors[ticker], linestyle=":", linewidth=1.2)
            mark_periods(ax, frame.index, splits)
            ax.set_title(ticker.upper(), pad=14)
            ax.set_ylabel(ylabel)
        for ax in axes[-1]:
            ax.set_xlabel("fecha")
        fig.suptitle(f"{title} ({', '.join(shown_periods(frame.index, splits))})", fontweight="bold")
        fig.legend(handles=[*extra_handles, boundary_handle()], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, filename)


def plot_log_prices(log_prices: pd.DataFrame, splits: dict) -> Path:
    """Figura 1: log-precio de cada uno de los 6 activos del grupo."""
    return asset_panels(log_prices, splits, "Figura 1 · Log-precios ajustados diarios del grupo elegido",
                        "log(precio en USD)", "fig1_log_precios.png")


def plot_beta(beta_t: pd.DataFrame, johansen_beta: dict, splits: dict) -> Path:
    """Figura 3: pesos β̂_t normalizados (una línea por activo) con el β de Johansen de train como referencia."""
    handles = (Line2D([], [], color=MUTED, linewidth=1.0, label="β̂_t del filtro de Kalman 1"),
               Line2D([], [], color=MUTED, linestyle=":", linewidth=1.2, label="β de Johansen (train)"))
    return asset_panels(beta_t, splits, "Figura 3 · Pesos β̂_t del filtro de Kalman 1 (normalizados)",
                        "peso sobre log-precio", "fig3_beta_t.png", reference=johansen_beta, extra_handles=handles)


def plot_rolling_eigenvector(eigvec: pd.DataFrame, splits: dict, window_days: int) -> Path:
    """Figura 4: componentes del primer eigenvector de Johansen móvil (vector unitario)."""
    handles = (Line2D([], [], color=MUTED, linewidth=1.0, label=f"primer eigenvector, ventana de {window_days} días pasados"),)
    return asset_panels(eigvec, splits, "Figura 4 · Primer eigenvector de Johansen móvil en el tiempo",
                        "componente (vector unitario)", "fig4_eigenvector.png", extra_handles=handles)


def plot_filter_diagnostic(posvar: pd.DataFrame, gain: pd.DataFrame, burn_in_end: pd.Timestamp, band: np.ndarray,
                           colors: dict) -> Path:
    """Diagnóstico (solo train): varianza posterior P_t (escala log) y |K_t| (escala lineal) por componente."""
    names = list(posvar.columns)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, len(names), figsize=(3.3 * len(names), 6.5), sharex=True)
        for i, name in enumerate(names):
            top, bottom = axes[0, i], axes[1, i]
            top.axhspan(band[0, i], band[1, i], color=SHADE, linewidth=0)
            top.plot(posvar.index, posvar[name], color=colors[name], linewidth=1.0)
            top.set_yscale("log")
            top.set_title(f"β {name.upper()}")
            after = gain.loc[gain.index > burn_in_end, name].abs()
            bottom.plot(gain.index, gain[name].abs(), color=colors[name], linewidth=0.6)
            bottom.set_ylim(0, 1.15 * after.max())
            bottom.text(0.98, 0.97, f"máx. en burn-in: {gain[name].abs().max():.2g}", transform=bottom.transAxes,
                        ha="right", va="top", fontsize=7, color=MUTED)
            for ax in (top, bottom):
                ax.axvline(burn_in_end, color=MUTED, linestyle="--", linewidth=0.9)
                ax.xaxis.set_major_locator(mdates.YearLocator(2))
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            bottom.set_xlabel("fecha")
        axes[0, 0].set_ylabel("P_t|t (escala log)")
        axes[1, 0].set_ylabel("|K_t| (escala lineal)")
        fig.suptitle("Diagnóstico del filtro de Kalman 1 (train): P_t baja hasta su piso y K_t se estabiliza",
                     fontweight="bold")
        fig.legend(handles=[Patch(color=SHADE, label="banda estable de P_t (train)"),
                            Line2D([], [], color=MUTED, linestyle="--", linewidth=0.9, label="fin del burn-in")],
                   loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.03))
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return save(fig, "fig_diagnostico_filtro1.png")
