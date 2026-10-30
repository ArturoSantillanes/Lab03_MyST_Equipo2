"""Etapa 'cointegration': embudo de 7 etapas en cada sector y selección final del grupo y su vector β (solo train).

Dueño: Milca
Lee: data/processed/prices.csv, data/processed/splits.json, data/processed/sectors.json
Escribe: artifacts/cointegration/group_selection.csv (todos los subgrupos de todos los sectores),
         candidatos_final.csv (los siete requisitos de cada grupo evaluado), q_grids_candidatos.csv (rejilla de q de
         cada candidato), selected_group.json (grupo ganador y conteo de pruebas), stability_windows.csv (etapa 7 del
         ganador), eigvec_rolling.csv, y las figuras 1 y 4 en artifacts/figures/
Reglas: Todo lo que se estima o elige usa SOLO train y log-precios. Contar TODO lo que se prueba.
El eigenvector móvil se calcula hacia adelante en toda la muestra con solo datos pasados (Arturo y Paula lo usan);
ninguna cifra de test o valid se usa para decidir.
"""
import json
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration.figures import plot_log_prices, plot_rolling_eigenvector
from cointegration.final_selection import RELEVANCE_SHARE, describe_selection, run_final_selection
from cointegration.group_search import run_funnel
from cointegration.rolling_eigvec import WINDOW as EIGVEC_WINDOW
from cointegration.rolling_eigvec import rolling_first_eigenvector
from cointegration.universe import load_universe
from data.split import assign_period, slice_period

ROOT = Path(__file__).resolve().parent.parent
PRICES_PATH = ROOT / "data" / "processed" / "prices.csv"
SPLITS_PATH = ROOT / "data" / "processed" / "splits.json"
SECTORS_PATH = ROOT / "data" / "processed" / "sectors.json"
OUT_DIR = ROOT / "artifacts" / "cointegration"
SIGNIFICANCE = 0.05


def q_grids_table(results: dict) -> pd.DataFrame:
    """Rejilla de q (fina y, si se refinó, ultrafina) de cada candidato con filtro 1."""
    grids = [r["calibration"]["grid"].assign(group=g) for g, r in results.items() if r["calibration"] is not None]
    return pd.concat(grids, ignore_index=True)[["group", "resolution", *grids[0].columns.drop(["group", "resolution"])]]


def multiple_testing(membership: dict, funnel: pd.DataFrame, candidates: pd.DataFrame, results: dict,
                     q_grids: pd.DataFrame, history: int) -> dict:
    """Conteo exacto de sectores, grupos y pruebas estadísticas, con falsos positivos esperados y Bonferroni."""
    n_johansen = len(candidates)
    windows = sum(len(r["windows"]) for r in results.values())
    adf_columns = sum(len(g.split("-")) * (1 + len(r["windows"])) for g, r in results.items())
    groups_v5 = len(funnel) + int((candidates["origin"] != "embudo del sector").sum())
    return {
        "sectors": len(membership), "assets": sum(len(t) for t in membership.values()),
        "groups_stage2_correlation": len(funnel), "trimmed_groups": groups_v5 - len(funnel), "groups_v5": groups_v5,
        "groups_before_v5": history, "groups_all_versions": history + groups_v5,
        "engle_granger_pair_tests": sum(comb(len(t), 2) for t in membership.values()),
        "johansen_tests_train": n_johansen, "johansen_tests_subwindows": windows,
        "adf_tests_johansen_columns": adf_columns, "q_configurations": len(q_grids), "adf_tests_kalman": len(q_grids),
        "expected_false_positives_johansen_5pct": SIGNIFICANCE * n_johansen,
        "bonferroni_threshold_johansen": SIGNIFICANCE / n_johansen,
    }


def run() -> None:
    """Embudo por sector y selección final en train; guarda tablas, grupo ganador, estabilidad, eigenvector y figuras."""
    log_prices = np.log(pd.read_csv(PRICES_PATH, index_col="date", parse_dates=True).drop(columns="period"))
    splits = json.loads(SPLITS_PATH.read_text())
    membership = {s: m["tickers"] for s, m in json.loads(SECTORS_PATH.read_text()).items()}
    train = slice_period(log_prices, splits, "train")

    tables, funnel_details = [], {}
    for sector, tickers in membership.items():
        table, details = run_funnel(train[tickers], sector)
        tables.append(table)
        funnel_details.update(details)
        print(f"  {sector}: {len(table)} subgrupos, {int(table['stage2_pass'].sum())} pasan correlación, "
              f"{int((table['stage_reached'] >= 4).sum())} llegan a Johansen, {int((table['stage_reached'] == 7).sum())} cumplen 4 a 6")
    funnel = pd.concat(tables, ignore_index=True)
    candidates, winner, details, results = run_final_selection(train, membership, funnel_details)
    if not winner["R2"] or details["stage"]["chosen_column"] is None or details["calibration"]["q"] is None:
        raise RuntimeError("El mejor grupo no tiene vector válido o q válido; ver candidatos_final.csv (no se relaja nada)")

    q_grids = q_grids_table(results)
    history = load_universe()["preregistration"]["previously_tested"]["historial_total_grupos_antes_de_v5"]
    selection = describe_selection(details, winner, train)
    tickers = selection["tickers"]
    selection.update(
        relevance_share_threshold=RELEVANCE_SHARE,
        n_candidates=len(candidates), n_eligible=int(candidates["eligible"].sum()),
        stage2_fallback_sectors=sorted(funnel.loc[funnel["stage2_fallback_used"], "sector"].unique().tolist()),
        fallback_count=int(funnel.groupby("sector")["stage2_fallback_used"].first().sum()),
        multiple_testing=multiple_testing(membership, funnel, candidates, results, q_grids, history),
    )
    eigvec = rolling_first_eigenvector(log_prices[tickers], selection["var_order"]["used"],
                                       np.array([selection["beta"][t] for t in tickers]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    funnel.to_csv(OUT_DIR / "group_selection.csv", index=False)
    candidates.to_csv(OUT_DIR / "candidatos_final.csv", index=False)
    q_grids.to_csv(OUT_DIR / "q_grids_candidatos.csv", index=False)
    (OUT_DIR / "selected_group.json").write_text(json.dumps(selection, indent=2, ensure_ascii=False))
    details["windows"].to_csv(OUT_DIR / "stability_windows.csv", index=False)
    eigvec.assign(period=assign_period(eigvec.index, splits)).to_csv(OUT_DIR / "eigvec_rolling.csv", date_format="%Y-%m-%d")
    plot_log_prices(log_prices[tickers], splits)
    plot_rolling_eigenvector(eigvec, splits, EIGVEC_WINDOW)

    mt = selection["multiple_testing"]
    print(f"  selección final: {len(candidates)} grupos evaluados, {selection['n_eligible']} elegibles; ganador {tickers} "
          f"({selection['sector']}, {selection['n_requirements']}/7 requisitos)")
    print(f"  ganador (train): r̂ = {selection['r_hat']} / {selection['r_hat_maxeig']}; ADF p = {selection['adf_pvalue']:.2e}; "
          f"vida media = {selection['half_life_days']:.1f} días; estabilidad {selection['stable_windows']}")
    print(f"  pruebas: {mt['groups_v5']} grupos en v5 ({mt['groups_all_versions']} con versiones anteriores), "
          f"{mt['johansen_tests_train']} Johansen en train, {mt['q_configurations']} configuraciones de q")
