"""Etapa 'cointegration': embudo de 7 etapas para elegir los 6 activos finales y su vector β (solo train).

Dueño: Milca
Lee: data/processed/prices.csv, data/processed/splits.json, data/universe.json
Escribe: artifacts/cointegration/group_selection.csv (los 84 subgrupos), selected_group.json,
         stability_windows.csv (etapa 7), eigvec_rolling.csv, y las figuras 1 y 4 en artifacts/figures/
Reglas: Todo lo que se estima o elige usa SOLO train y log-precios. Contar TODO lo que se prueba.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration.figures import plot_log_prices, plot_rolling_eigenvector
from cointegration.group_search import run_funnel
from cointegration.rolling_eigvec import WINDOW as EIGVEC_WINDOW
from cointegration.rolling_eigvec import rolling_first_eigenvector
from cointegration.stability import stability_windows
from data.split import assign_period, slice_period

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.json"
OUT_DIR = ROOT / "artifacts" / "cointegration"


def run() -> None:
    """Corre el embudo en train, guarda la tabla, el grupo elegido, la estabilidad y el eigenvector móvil."""
    log_prices = np.log(pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True).drop(columns="period"))
    splits = json.loads(SPLITS_PATH.read_text())
    train = slice_period(log_prices, splits, "train")

    table, selection = run_funnel(train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "group_selection.csv", index=False)
    if selection is None:
        raise RuntimeError("Ningún subgrupo cumple las etapas 2 a 6; ver group_selection.csv (no se relajan criterios)")

    tickers = selection["tickers"]
    stability = stability_windows(train[tickers], selection["var_order"]["used"])
    selection["stability_share_cointegrated"] = float(stability["still_cointegrated"].mean())
    selection["stability_windows"] = len(stability)
    beta = np.array([selection["beta"][t] for t in tickers])
    eigvec = rolling_first_eigenvector(log_prices[tickers], selection["var_order"]["used"], beta)

    (OUT_DIR / "selected_group.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    stability.to_csv(OUT_DIR / "stability_windows.csv", index=False)
    eigvec.assign(period=assign_period(eigvec.index, splits)).to_csv(OUT_DIR / "eigvec_rolling.csv", date_format="%Y-%m-%d")
    plot_log_prices(log_prices[tickers], splits)
    plot_rolling_eigenvector(eigvec, splits, EIGVEC_WINDOW)

    stages = table["stage_reached"].value_counts().sort_index().to_dict()
    print(f"  subgrupos evaluados: {len(table)}; etapa alcanzada: {stages}; llegaron a Johansen: "
          f"{selection['n_subgroups_reached_johansen']}")
    print(f"  elegido: {tickers}; r̂ = {selection['r_hat']} (traza) / {selection['r_hat_maxeig']} (máx. eig.); "
          f"ADF p = {selection['adf_pvalue']:.2e}; vida media = {selection['half_life_days']:.1f} días (train)")
    print(f"  estabilidad: cointegrado en {stability['still_cointegrated'].sum()} de {len(stability)} sub-ventanas (train)")
