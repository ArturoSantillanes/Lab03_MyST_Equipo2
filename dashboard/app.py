"""Dashboard interactivo (Streamlit) con vista ejecutiva y vista técnica.

Dueño: Paula
Lee: Archivos de artifacts/ (no re-corre el backtest).
Escribe: Nada.
Reglas: Un solo comando: streamlit run dashboard/app.py. Todas las gráficas interactivas.

Pestañas: vista ejecutiva (capital y drawdown, tarjeta de métricas, atribución de costos, bitácora y retornos por
operación), vista técnica (spread con señales, β̂_t, eigenvector móvil, ganancias y varianzas posteriores de ambos
filtros, z-score con umbrales y Sharpe móvil) y calibración (rejilla de train, banda y supuestos). La barra lateral
filtra por tramo y por rango de fechas; todas las gráficas son Plotly (zoom, hover y leyenda que oculta series).
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
PERIODS = ("train", "test", "valid")
PERIOD_COLORS = {"train": "#2a78d6", "test": "#eb6834", "valid": "#1baf7a"}
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NEEDED = ["backtest/equity.csv", "backtest/trades.csv", "backtest/metrics.json", "backtest/params.json",
          "backtest/positions.csv", "backtest/rolling_sharpe.csv", "backtest/grid_search.csv", "kalman/z_t.csv",
          "kalman/beta_t.csv", "kalman/gain_1.csv", "kalman/gain_2.csv", "kalman/posvar_1.csv", "kalman/posvar_2.csv",
          "cointegration/eigvec_rolling.csv"]


@st.cache_data
def dated(name: str) -> pd.DataFrame:
    """CSV de artifacts/ con la fecha como índice."""
    return pd.read_csv(ART / name, index_col="date", parse_dates=True)


@st.cache_data
def table(name: str) -> pd.DataFrame:
    """CSV de artifacts/ sin índice de fechas."""
    return pd.read_csv(ART / name)


@st.cache_data
def load_json(name: str) -> dict:
    """JSON de artifacts/ o data/."""
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def number(value, fmt: str = "{:.2f}") -> str:
    """Formatea métricas que pueden venir como None (NaN) o 'inf'."""
    if value is None:
        return "n/d"
    if value == "inf":
        return "∞"
    return fmt.format(value)


def add_bounds(fig: go.Figure, splits: dict, start: pd.Timestamp, end: pd.Timestamp, rows: int = 1) -> None:
    """Fronteras entre tramos (línea punteada) y nombre del tramo arriba, solo para las fechas visibles."""
    for period in PERIODS:
        lo, hi = max(pd.Timestamp(splits[period]["start"]), start), min(pd.Timestamp(splits[period]["end"]), end)
        if lo > hi:
            continue
        if period != "train" and pd.Timestamp(splits[period]["start"]) >= start:
            for row in range(1, rows + 1):
                fig.add_vline(x=pd.Timestamp(splits[period]["start"]), line_dash="dash", line_color="#52514e",
                              line_width=1, row=row if rows > 1 else None, col=1 if rows > 1 else None)
        fig.add_annotation(x=lo + (hi - lo) / 2, y=1.0, yref="paper", text=period, showarrow=False, yanchor="bottom",
                           font={"color": PERIOD_COLORS[period], "size": 12})


def style(fig: go.Figure, title: str, y_title: str, height: int = 420) -> go.Figure:
    """Formato común: título, ejes, leyenda horizontal y hover unificado."""
    fig.update_layout(title=title, height=height, hovermode="x unified", margin={"t": 70, "b": 40},
                      legend={"orientation": "h", "y": -0.22})
    fig.update_yaxes(title_text=y_title)
    fig.update_xaxes(title_text="fecha")
    return fig


def show(fig: go.Figure) -> None:
    """Muestra una figura Plotly a todo lo ancho."""
    st.plotly_chart(fig, width="stretch")


def executive_view(equity, trades, metrics, splits, periods, start, end):
    """Capital y drawdown, tarjeta de métricas, atribución de costos, bitácora e histograma."""
    st.subheader("Curva de capital y drawdown")
    st.caption("Cada tramo es una corrida independiente con parámetros congelados en train que empieza con "
               "$1,000,000. Equity neto de comisiones y préstamo.")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05)
    for period in periods:
        part = equity[equity["period"] == period]
        fig.add_trace(go.Scatter(x=part.index, y=part["equity"], name=f"equity {period}",
                                 line={"color": PERIOD_COLORS[period]}), row=1, col=1)
        fig.add_trace(go.Scatter(x=part.index, y=part["drawdown"] * 100, name=f"drawdown {period}", fill="tozeroy",
                                 line={"color": PERIOD_COLORS[period], "width": 0.5}, opacity=0.4), row=2, col=1)
    fig.add_hline(y=1_000_000, line_dash="dot", line_color="#52514e", row=1, col=1)
    add_bounds(fig, splits, start, end, rows=2)
    style(fig, "Equity (USD) con el drawdown (%) debajo", "", 560)
    fig.update_xaxes(title_text=None, row=1, col=1)
    fig.update_yaxes(title_text="equity (USD)", tickformat="$,.0f", row=1, col=1)
    fig.update_yaxes(title_text="drawdown (%)", row=2, col=1)
    show(fig)

    st.subheader("Tarjeta de métricas por tramo")
    for period in periods:
        m = metrics["by_period"][period]
        st.markdown(f"**{period}** · {m['start']} a {m['end']} · {m['n_trades']} operaciones · "
                    f"rendimiento total {m['total_return']:+.2%}")
        cols = st.columns(6)
        cols[0].metric("Sharpe", number(m["sharpe"]), border=True)
        cols[1].metric("Sortino", number(m["sortino"]), border=True)
        cols[2].metric("Calmar", number(m["calmar"]), border=True)
        cols[3].metric("Max drawdown", number(m["max_drawdown"], "{:.2%}"), border=True)
        cols[4].metric("Win rate", number(m["win_rate"], "{:.1%}"), border=True)
        cols[5].metric("Profit factor", number(m["profit_factor"]), border=True)
    st.caption("Tasa libre de riesgo 0 y anualización con 252 días (ver performance/metrics.py).")

    st.subheader("Atribución de costos: del P&L bruto al neto")
    cols = st.columns(len(periods)) if periods else []
    for col, period in zip(cols, periods):
        a = metrics["by_period"][period]["pnl_attribution"]
        fig = go.Figure(go.Waterfall(
            x=["P&L bruto", "comisiones", "préstamo", "P&L neto"], measure=["absolute", "relative", "relative", "total"],
            y=[a["gross_pnl"], -a["commission"], -a["borrow_cost"], a["net_pnl"]],
            text=[f"${v:,.0f}" for v in (a["gross_pnl"], -a["commission"], -a["borrow_cost"], a["net_pnl"])],
            connector={"line": {"color": "#d9d8d3"}}))
        fig.update_layout(title=f"{period}", height=360, margin={"t": 50, "b": 30}, yaxis_tickformat="$,.0f",
                          yaxis_title="USD", showlegend=False)
        with col:
            show(fig)

    st.subheader("Bitácora de operaciones")
    c1, c2 = st.columns(2)
    direction = c1.multiselect("Dirección", ["largo", "corto"], default=["largo", "corto"])
    outcome = c2.multiselect("Resultado", ["ganadora", "perdedora"], default=["ganadora", "perdedora"])
    log = trades[trades["period"].isin(periods) & trades["direction"].isin(direction)].copy()
    log["resultado"] = np.where(log["net_pnl"] > 0, "ganadora", "perdedora")
    log = log[log["resultado"].isin(outcome)]
    log = log[(log["exit_date"] >= start) & (log["entry_date"] <= end)]
    st.dataframe(log[["entry_date", "exit_date", "period", "direction", "gross_pnl", "commission", "borrow_cost",
                      "net_pnl", "return", "holding_days", "exit_reason", "resultado"]], hide_index=True,
                 column_config={"entry_date": st.column_config.DateColumn("entrada"),
                                "exit_date": st.column_config.DateColumn("salida"),
                                "gross_pnl": st.column_config.NumberColumn("P&L bruto", format="$%.0f"),
                                "commission": st.column_config.NumberColumn("comisiones", format="$%.0f"),
                                "borrow_cost": st.column_config.NumberColumn("préstamo", format="$%.0f"),
                                "net_pnl": st.column_config.NumberColumn("P&L neto", format="$%.0f"),
                                "return": st.column_config.NumberColumn("retorno", format="percent"),
                                "holding_days": st.column_config.NumberColumn("días")})
    st.caption(f"{len(log)} operaciones con los filtros actuales; P&L neto total ${log['net_pnl'].sum():,.0f}.")

    st.subheader("Distribución de retornos por operación")
    values = log["return"] * 100
    fig = go.Figure()
    if len(values):
        low, high = float(values.min()), float(values.max())
        size = (high - low) / 20 or 0.1
        for period in periods:
            fig.add_trace(go.Histogram(x=values[log["period"] == period], name=period, opacity=0.65,
                                       xbins={"start": low, "end": high + size, "size": size},
                                       marker_color=PERIOD_COLORS[period]))
    if len(values) > 1:
        grid = np.linspace(low, high, 200)
        mu, sigma = values.mean(), values.std(ddof=1)
        density = np.exp(-0.5 * ((grid - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
        fig.add_trace(go.Scatter(x=grid, y=density * len(values) * size, name=f"normal (μ={mu:.2f}%, σ={sigma:.2f}%)",
                                 line={"color": "#0b0b0b"}))
    fig.update_layout(barmode="stack", height=400, title="Retorno neto por operación (% del equity de entrada) de "
                      "las operaciones filtradas, con una normal ajustada", xaxis_title="retorno neto (%)",
                      yaxis_title="número de operaciones", legend={"orientation": "h", "y": -0.2})
    show(fig)


def line_panel(frame: pd.DataFrame, columns: list, title: str, y_title: str, splits, start, end, log_y=False,
               settled: pd.Series | None = None):
    """Una línea por columna con fronteras entre tramos.

    settled marca las filas fuera del burn-in: el eje y se ajusta a ellas para que el arranque del filtro (P₀ difuso)
    no aplaste la escala; el burn-in sigue en la gráfica y se ve con el zoom.
    """
    fig = go.Figure()
    for i, column in enumerate(columns):
        fig.add_trace(go.Scatter(x=frame.index, y=frame[column], name=column, mode="lines",
                                 line={"color": SERIES_COLORS[i % len(SERIES_COLORS)], "width": 1.2}))
    add_bounds(fig, splits, start, end)
    style(fig, title, y_title)
    values = frame.loc[settled.reindex(frame.index, fill_value=False), columns] if settled is not None else None
    if log_y:
        fig.update_yaxes(type="log")
    if values is not None and values.notna().any().any():
        low, high = float(np.nanmin(values.to_numpy())), float(np.nanmax(values.to_numpy()))
        if log_y:
            fig.update_yaxes(range=[math.log10(low) - 0.3, math.log10(high) + 0.3])
        else:
            pad = 0.1 * (high - low or abs(high) or 1.0)
            fig.update_yaxes(range=[low - pad, high + pad])
    return fig


def technical_view(positions, params, splits, start, end, window):
    """Spread con señales, β̂_t, eigenvector, ganancias, z con umbrales, varianzas posteriores y Sharpe móvil."""
    z_t = window(dated("kalman/z_t.csv"))
    pos = window(positions)
    st.subheader("Spread S_t con señales de entrada y salida")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_t.index, y=z_t["spread"], name="spread S_t (filtro 1)", line={"color": "#2a78d6"}))
    fig.add_trace(go.Scatter(x=z_t.index, y=z_t["mu_prev"], name="media previa μ̂ (filtro 2)", line={"color": "#0b0b0b"}))
    previous = pos["position"].shift(fill_value=0)
    entries = {"entrada larga": (previous == 0) & (pos["position"] == 1),
               "entrada corta": (previous == 0) & (pos["position"] == -1)}
    markers = {"entrada larga": ("triangle-up", "#1baf7a"), "entrada corta": ("triangle-down", "#eb6834")}
    for label, mask in entries.items():
        days = pos.index[mask]
        fig.add_trace(go.Scatter(x=days, y=z_t.loc[days, "spread"], mode="markers", name=label,
                                 marker={"symbol": markers[label][0], "size": 10, "color": markers[label][1]}))
    exits = pos[pos["exit_reason"].fillna("") != ""]
    for reason, symbol in {"senal": "circle-open", "stop": "x", "tenencia_maxima": "square",
                           "fin_de_periodo": "diamond-open"}.items():
        days = exits.index[exits["exit_reason"] == reason]
        if len(days):
            fig.add_trace(go.Scatter(x=days, y=z_t.loc[days, "spread"], mode="markers", name=f"salida: {reason}",
                                     marker={"symbol": symbol, "size": 9, "color": "#0b0b0b"}))
    add_bounds(fig, splits, start, end)
    show(style(fig, "Spread S_t = β̂_tᵀ(log P_t − media de train) con entradas y salidas (parámetros finales)",
               "spread (log)", 460))

    st.subheader("Z-score con las líneas de umbral")
    fig = go.Figure(go.Scatter(x=z_t.index, y=z_t["z"], name="z_t", line={"color": "#2a78d6", "width": 1}))
    for key, color in (("z_entry", "#eb6834"), ("z_exit", "#1baf7a"), ("z_stop", "#0b0b0b")):
        value = params[key]
        if value == "inf":
            continue
        for sign in (1, -1):
            fig.add_trace(go.Scatter(x=[z_t.index.min(), z_t.index.max()], y=[sign * value] * 2, mode="lines",
                                     name=f"±{key} = {value}", legendgroup=key, showlegend=sign > 0,
                                     line={"color": color, "dash": "dash", "width": 1}, hoverinfo="skip"))
    add_bounds(fig, splits, start, end)
    show(style(fig, f"z_t (filtro 2, estadísticos previos); umbrales finales de train · max_hold = "
                    f"{params['max_hold_days']} días", "z"))

    st.subheader("Pesos dinámicos β̂_t por activo")
    beta = window(dated("kalman/beta_t.csv"))
    tickers = [c for c in beta.columns if c not in ("period", "burn_in")]
    show(line_panel(beta, tickers, "β̂_t del filtro 1 (normalizado con xel = +1; eje ajustado fuera del burn-in)",
                    "β̂_t", splits, start, end, settled=~beta["burn_in"].astype(bool)))

    st.subheader("Primer eigenvector de Johansen móvil")
    eig = window(dated("cointegration/eigvec_rolling.csv"))
    show(line_panel(eig, tickers, "Primer eigenvector de Johansen (ventana móvil de 504 días pasados, vector unitario)",
                    "componente", splits, start, end))
    st.caption("Los saltos son cambios de signo y de vector dominante de una ventana a otra: con 504 días el "
               "eigenvector no está bien identificado (ver handoff de Milca); por eso se opera con β̂_t de Kalman.")

    st.subheader("Ganancia de Kalman de ambos filtros")
    gain_1, gain_2 = window(dated("kalman/gain_1.csv")), window(dated("kalman/gain_2.csv"))
    c1, c2 = st.columns(2)
    filter1_assets = [c for c in gain_1.columns if c not in ("period", "burn_in")]
    with c1:
        show(line_panel(gain_1[filter1_assets].abs(), filter1_assets,
                        "Filtro 1: |K_t| por activo (xel es la pata de normalización, no es estado)", "|K_t|",
                        splits, start, end, settled=~gain_1["burn_in"].astype(bool)))
    with c2:
        show(line_panel(gain_2[["mu", "sigma2"]].abs(), ["mu", "sigma2"], "Filtro 2: |K_t| de μ y de σ²", "|K_t|",
                        splits, start, end, settled=~gain_2["burn_in"].astype(bool)))

    st.subheader("Varianza posterior de ambos filtros")
    posvar_1, posvar_2 = window(dated("kalman/posvar_1.csv")), window(dated("kalman/posvar_2.csv"))
    c1, c2 = st.columns(2)
    with c1:
        show(line_panel(posvar_1, filter1_assets, "Filtro 1: diagonal de P_t|t (escala log)", "P_t|t", splits, start,
                        end, log_y=True, settled=~posvar_1["burn_in"].astype(bool)))
    with c2:
        show(line_panel(posvar_2, ["mu", "sigma2"], "Filtro 2: P_t|t de μ y de σ² (escala log)", "P_t|t", splits,
                        start, end, log_y=True, settled=~posvar_2["burn_in"].astype(bool)))
    st.caption("Los ejes de ganancias y varianzas se ajustan al tramo sin burn-in; el arranque de los filtros (P₀ "
               "difuso) sigue en la gráfica: use el zoom o el botón de autoescala para verlo.")

    st.subheader("Sharpe móvil de 252 días")
    rolling = window(dated("backtest/rolling_sharpe.csv"))
    fig = go.Figure(go.Scatter(x=rolling.index, y=rolling["rolling_sharpe_252"], name="Sharpe 252 días",
                               customdata=rolling["window_periods"], line={"color": "#2a78d6"},
                               hovertemplate="%{y:.2f} (ventana: %{customdata})"))
    fig.add_hline(y=0, line_color="#52514e", line_dash="dot")
    add_bounds(fig, splits, start, end)
    show(style(fig, "Sharpe anualizado móvil (252 días, tramos encadenados)", "Sharpe"))


def calibration_view(grid, params, metrics, band_grid, comparison):
    """Rejilla de train (meseta o pico), banda de rebalanceo, DSR y supuestos."""
    sel = params["selection"]
    st.subheader("Grid search (solo train)")
    st.markdown(f"**{params['grid']['n_configurations']:,} configuraciones**, {params['grid']['n_viable']} con al menos "
                f"{params['n_min_trades']} operaciones (las demás valen −1e9). Elegido: z_entry = {params['z_entry']}, "
                f"z_exit = {params['z_exit']}, z_stop = {params['z_stop']}, max_hold = {params['max_hold_days']} "
                f"(Sharpe de train {sel['chosen_sharpe_train']:.2f}, {sel['chosen_n_trades_train']} operaciones). "
                f"Mejor bruto: Sharpe {sel['best_raw']['sharpe']:.2f} con {sel['best_raw']['n_trades']} operaciones "
                f"({'meseta' if sel['best_raw_is_plateau'] else 'pico aislado'}). Parámetros en la orilla de la rejilla: "
                f"{', '.join(k for k, v in sel['on_edge'].items() if v) or 'ninguno'}.")
    c1, c2 = st.columns(2)
    stop = c1.selectbox("z_stop", sorted(grid["z_stop"].unique()), index=sorted(grid["z_stop"].unique()).index(
        float(params["z_stop"])) if params["z_stop"] != "inf" else len(grid["z_stop"].unique()) - 1)
    hold = c2.selectbox("max_hold_days", sorted(grid["max_hold_days"].unique()),
                        index=sorted(grid["max_hold_days"].unique()).index(params["max_hold_days"]))
    part = grid[(grid["z_stop"] == stop) & (grid["max_hold_days"] == hold)]
    sharpe = part.pivot(index="z_entry", columns="z_exit", values="sharpe")
    trades = part.pivot(index="z_entry", columns="z_exit", values="n_trades")
    text = sharpe.round(2).astype(str) + "<br>" + trades.astype("Int64").astype(str) + " ops"
    fig = go.Figure(go.Heatmap(z=sharpe.values, x=[str(c) for c in sharpe.columns], y=[str(i) for i in sharpe.index],
                               text=text.values, texttemplate="%{text}", colorscale="RdBu", zmid=0,
                               colorbar={"title": "Sharpe"}))
    fig.update_layout(title=f"Sharpe de train por z_entry × z_exit (z_stop = {stop}, max_hold = {hold}); "
                            f"debajo, número de operaciones", height=520, xaxis_title="z_exit", yaxis_title="z_entry",
                      xaxis_type="category", yaxis_type="category")
    show(fig)
    st.caption("Las celdas con pocas operaciones pueden tener Sharpe alto: es el problema de tamaño de muestra que "
               "corrige la restricción de operaciones mínimas.")

    st.subheader("Rebalanceo diario contra banda")
    st.dataframe(comparison, hide_index=True)
    fig = go.Figure(go.Scatter(x=band_grid["band"].astype(str), y=band_grid["sharpe"], mode="lines+markers"))
    fig.update_layout(title="Sharpe de train por banda de no-rebalanceo (umbrales congelados; inf = nunca "
                            "rebalancear salvo por el tope de 80%)", height=360, xaxis_type="category",
                      xaxis_title="banda (fracción del nocional objetivo)", yaxis_title="Sharpe de train")
    show(fig)

    st.subheader("Deflated Sharpe Ratio (train)")
    dsr = metrics["deflated_sharpe"]
    st.dataframe(pd.DataFrame(dsr["train"]).T[["n_trials", "sr_daily", "sr0_annualized", "dsr"]],
                 column_config={"dsr": st.column_config.NumberColumn("DSR", format="%.2e")})
    psr = {p: v["dsr"] for p, v in dsr["out_of_sample_psr"].items()}
    st.markdown(f"Fuera de muestra (una sola prueba por tramo, PSR contra Sharpe 0): test {psr['test']:.2f}, "
                f"valid {psr['valid']:.2f}. Pruebas de Johansen en train: {dsr['johansen_tests_train']} "
                f"({dsr['expected_false_positives_johansen_5pct']} falsos positivos esperados al 5%).")
    st.subheader("Supuestos")
    st.json(params["assumptions"])
    st.markdown(f"Valid evaluada **{params['validation_evaluations']}** vez con parámetros congelados.")


def main() -> None:
    """Arma la página: barra lateral con filtros y tres pestañas."""
    st.set_page_config(page_title="Lab 03 · Arbitraje estadístico", layout="wide")
    st.title("Lab 03 · Arbitraje estadístico con cointegración y filtros de Kalman")
    missing = [name for name in NEEDED if not (ART / name).exists()]
    if missing:
        st.error(f"Faltan artefactos: {missing}. Corre primero `python main.py`.")
        st.stop()
    splits = load_json("data/processed/splits.json")
    params = load_json("artifacts/backtest/params.json")
    metrics = load_json("artifacts/backtest/metrics.json")
    selection = load_json("artifacts/cointegration/selected_group.json")
    equity = dated("backtest/equity.csv")
    trades = table("backtest/trades.csv").astype({"entry_date": "datetime64[ns]", "exit_date": "datetime64[ns]"})
    positions = dated("backtest/positions.csv")

    st.sidebar.header("Filtros")
    periods = st.sidebar.multiselect("Tramos", list(PERIODS), default=list(PERIODS))
    first, last = pd.Timestamp(splits["train"]["start"]).date(), pd.Timestamp(splits["valid"]["end"]).date()
    dates = st.sidebar.date_input("Rango de fechas", (first, last), min_value=first, max_value=last)
    start, end = (pd.Timestamp(dates[0]), pd.Timestamp(dates[1])) if len(dates) == 2 else (pd.Timestamp(first),
                                                                                          pd.Timestamp(last))
    st.sidebar.markdown(f"**Grupo:** {', '.join(t.upper() for t in selection['tickers'])}  \n"
                        f"**Umbrales (train):** entrada {params['z_entry']}, salida {params['z_exit']}, stop "
                        f"{params['z_stop']}, tenencia máx. {params['max_hold_days']} días  \n"
                        f"**Política:** {params['policy']} (banda {params['band']})")

    def window(frame: pd.DataFrame) -> pd.DataFrame:
        """Filtra por rango de fechas y por los tramos elegidos."""
        out = frame.loc[start:end]
        return out[out["period"].isin(periods)] if "period" in out.columns else out

    executive, technical, calibration = st.tabs(["Vista ejecutiva", "Vista técnica", "Calibración y supuestos"])
    with executive:
        executive_view(window(equity), trades, metrics, splits, periods, start, end)
    with technical:
        technical_view(positions, params, splits, start, end, window)
    with calibration:
        calibration_view(table("backtest/grid_search.csv"), params, metrics, table("backtest/band_grid.csv"),
                         table("backtest/rebalance_comparison.csv"))


main()
