"""Etapa 'cointegration': embudo de 7 etapas y selección final del grupo y su vector β (solo train).

Dueño: Milca
Lee: data/processed/prices.csv, data/processed/splits.json, data/universe.json
Escribe: artifacts/cointegration/group_selection.csv (los 84 subgrupos), candidatos_final.csv (selección final),
         selected_group.json (grupo ganador), stability_windows.csv (etapa 7 del ganador), eigvec_rolling.csv,
         y las figuras 1 y 4 en artifacts/figures/
Reglas: Todo lo que se estima o elige usa SOLO train y log-precios. Contar TODO lo que se prueba.
El eigenvector móvil se calcula hacia adelante en toda la muestra con solo datos pasados (Arturo y Paula lo usan);
ninguna cifra de test o valid se usa para decidir.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration.figures import plot_log_prices, plot_rolling_eigenvector
from cointegration.final_selection import RELEVANCE_SHARE, notional_shares, run_final_selection
from cointegration.group_search import describe_selection, run_funnel
from cointegration.rolling_eigvec import WINDOW as EIGVEC_WINDOW
from cointegration.rolling_eigvec import rolling_first_eigenvector
from data.split import assign_period, slice_period

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.json"
OUT_DIR = ROOT / "artifacts" / "cointegration"


def run() -> None:
    """Embudo y selección final en train; guarda tablas, grupo ganador, estabilidad, eigenvector móvil y figuras."""
    log_prices = np.log(pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True).drop(columns="period"))
    splits = json.loads(SPLITS_PATH.read_text())
    train = slice_period(log_prices, splits, "train")

    table, funnel_selection = run_funnel(train)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "group_selection.csv", index=False)
    if funnel_selection is None:
        raise RuntimeError("Ningún subgrupo cumple las etapas 2 a 6; ver group_selection.csv (no se relajan criterios)")

    candidates, winner, details = run_final_selection(train, funnel_selection, table)
    candidates.to_csv(OUT_DIR / "candidatos_final.csv", index=False)
    if winner is None:
        raise RuntimeError("Ningún candidato es elegible; ver candidatos_final.csv (no se relajan criterios)")

    selection = describe_selection(details["stage"], train, len(table), funnel_selection["n_subgroups_reached_johansen"],
                                   funnel_selection["stage2_fallback_used"])
    tickers = selection["tickers"]
    selection.update(
        funnel_choice=funnel_selection["tickers"],
        funnel_choice_notional_share=notional_shares(funnel_selection["beta"]),
        relevance_share_threshold=RELEVANCE_SHARE,
        n_final_candidates=len(candidates),
        final_selection_rule=("elegible: r̂>=1 con traza y máx. eigenvalor al 95%, ADF p<0.05, vida media 5-60 días y un q "
                              "con cero cambios de signo; orden: estabilidad, ADF más negativo, más activos"),
        stability_share=float(winner["stability_share"]), stable_windows=winner["stable_windows"],
        notional_share=notional_shares(selection["beta"]),
    )
    eigvec = rolling_first_eigenvector(log_prices[tickers], selection["var_order"]["used"],
                                       np.array([selection["beta"][t] for t in tickers]))

    (OUT_DIR / "selected_group.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    details["windows"].to_csv(OUT_DIR / "stability_windows.csv", index=False)
    eigvec.assign(period=assign_period(eigvec.index, splits)).to_csv(OUT_DIR / "eigvec_rolling.csv", date_format="%Y-%m-%d")
    plot_log_prices(log_prices[tickers], splits)
    plot_rolling_eigenvector(eigvec, splits, EIGVEC_WINDOW)

    stages = table["stage_reached"].value_counts().sort_index().to_dict()
    print(f"  embudo: {len(table)} subgrupos; etapa alcanzada {stages}; {selection['n_subgroups_reached_johansen']} "
          f"llegaron a Johansen; el embudo eligió {funnel_selection['tickers']}")
    print(f"  selección final: {len(candidates)} candidatos, {int(candidates['eligible'].sum())} elegibles; ganador {tickers}")
    print(f"  ganador (train): r̂ = {selection['r_hat']} / {selection['r_hat_maxeig']}; ADF p = {selection['adf_pvalue']:.2e}; "
          f"vida media = {selection['half_life_days']:.1f} días; estabilidad {selection['stable_windows']}")
