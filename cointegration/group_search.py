"""Embudo de selección (etapas 2 a 7) de un sector sobre todos sus subgrupos de GROUP_SIZE activos, solo con train.

Dueño: Milca
Lee: log-precios de train de los activos de un sector.
Escribe: (en memoria) una fila por subgrupo con etapa alcanzada, estadísticos, decisión y razón, y los detalles
(Johansen, columnas, sub-ventanas, calibración del filtro 1) de cada subgrupo que llega a Johansen.
Reglas: SOLO train. Contar todo lo que se prueba. Protocolo v5, fijado en data/universe.json antes de descargar.

Etapa 2: correlación de retornos > 0.7 en las 3 sub-ventanas (heurística).
Etapa 3: Engle-Granger por pares: PRIORIZA (orden de evaluación), no elimina.
Etapas 4 a 6: Johansen, selección y validación de la columna para TODOS los subgrupos que pasan la etapa 2.
Respaldo de la etapa 2 (contado como respaldo): si en un sector ningún subgrupo la pasa, todos llegan a Johansen,
porque la correlación es una heurística y el criterio definitivo es Johansen.
Etapa 7: estabilidad en sub-ventanas para todo subgrupo que llega a Johansen. A los que cumplen 4 a 6 se les
calibra además el filtro 1 (rejilla fina de q, meseta, cambios de signo y burn-in).
"""
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from cointegration.correlation import (mean_level_correlation, passes_correlation, rolling_pair_correlations,
                                       subwindow_mean_correlations)
from cointegration.engle_granger import group_pair_stats, pair_pvalues, priority_order
from cointegration.johansen import DETERMINISTIC, johansen_test, rank_alarm, var_lag_orders
from cointegration.spread_checks import HALF_LIFE_RANGE, choose_column, column_table, normalize
from cointegration.stability import stability_share, stability_windows
import kalman.filter1_weights as filter1  # módulo completo: evita la importación circular cointegration <-> kalman

GROUP_SIZE = 6
SIGN_CONVENTION = ("largo el spread = comprar cada activo con β_i > 0 y vender cada activo con β_i < 0, en proporción "
                   "a |β_i|; corto el spread = todos los signos al revés")


def correlation_stage(train: pd.DataFrame) -> pd.DataFrame:
    """Etapa 2 para los C(n, 6) subgrupos del sector: correlación media por sub-ventana y si pasa."""
    pair_corr = rolling_pair_correlations(train)
    rows = []
    for group in itertools.combinations(train.columns, GROUP_SIZE):
        corr = subwindow_mean_correlations(group, pair_corr)
        rows.append({"subgroup": "-".join(group), **{f"corr_subwindow_{i + 1}": c for i, c in enumerate(corr)},
                     "corr_min_subwindow": min(corr), "stage2_pass": passes_correlation(corr),
                     "info_corr_log_levels": mean_level_correlation(group, train)})
    return pd.DataFrame(rows)


def stage2_fallback_used(stage2: pd.DataFrame) -> bool:
    """True si ningún subgrupo del sector pasó la heurística de correlación y todos deben llegar a Johansen."""
    return not stage2["stage2_pass"].any()


def engle_granger_stage(train: pd.DataFrame, stage2: pd.DataFrame) -> pd.DataFrame:
    """Etapa 3: estadísticos por pares y orden de prioridad (de los que pasaron la etapa 2, o de todos si hay respaldo)."""
    candidates = stage2 if stage2_fallback_used(stage2) else stage2[stage2["stage2_pass"]]
    pvalues = pair_pvalues(train)
    stats = pd.DataFrame([{"subgroup": s, **group_pair_stats(pvalues, tuple(s.split("-")))} for s in candidates["subgroup"]])
    return priority_order(stats)


def johansen_stage(prices: pd.DataFrame) -> dict:
    """Etapas 4, 5 y 6 para un subgrupo: Johansen, tabla de columnas, columna elegida y validación."""
    lags = var_lag_orders(prices)
    johansen = johansen_test(prices, lags["bic"])
    rank = johansen["rank_trace"]
    alarm = rank_alarm(rank, prices.shape[1])
    columns = column_table(prices, johansen["eigenvectors"], rank)
    chosen = None if alarm else choose_column(columns)
    return {"subgroup": "-".join(prices.columns), "lags": lags, "johansen": johansen, "alarm": alarm,
            "columns": columns, "chosen_column": chosen}


def johansen_row(stage: dict) -> dict:
    """Columnas de la tabla para un subgrupo que llegó a Johansen, con su decisión."""
    j = stage["johansen"]
    row = {"var_order_aic": stage["lags"]["aic"], "var_order_bic": stage["lags"]["bic"], "deterministic": DETERMINISTIC,
           "rank_trace": j["rank_trace"], "rank_maxeig": j["rank_maxeig"],
           "trace_r0": j["trace_stats"][0], "trace_crit95_r0": j["trace_crit_95"][0],
           "maxeig_r0": j["maxeig_stats"][0], "maxeig_crit95_r0": j["maxeig_crit_95"][0]}
    if stage["alarm"]:
        return {**row, "stage_reached": 4, "decision": "rechazado en Johansen", "reason": stage["alarm"]}
    if stage["chosen_column"] is None:
        return {**row, "stage_reached": 6, "decision": "rechazado en validación del spread",
                "reason": f"ninguna columna dentro de r̂ con ADF p < 0.05 y vida media en {HALF_LIFE_RANGE} días"}
    best = stage["columns"].iloc[stage["chosen_column"]]
    return {**row, "chosen_column": int(best["column"]), "adf_stat": best["adf_stat"], "adf_pvalue": best["adf_pvalue"],
            "half_life_days": best["half_life_days"], "stage_reached": 7, "decision": "cumple etapas 4 a 6",
            "reason": "Johansen con r̂ válido y spread operable; pasa a estabilidad y filtro 1"}


def evaluate_group(prices: pd.DataFrame) -> dict:
    """Etapas 4 a 7 y calibración del filtro 1 (si cumple 4 a 6) de un subgrupo; train solamente."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage = johansen_stage(prices)
        windows = stability_windows(prices, stage["johansen"]["var_order"])
        result = {"stage": stage, "windows": windows, "stability_share": stability_share(windows),
                  "stable_windows": f"{int(windows['stable'].sum())}/{len(windows)}", "beta": None, "calibration": None}
        if stage["chosen_column"] is not None:
            beta, leg = normalize(stage["johansen"]["eigenvectors"][:, stage["chosen_column"]])
            result["beta"] = dict(zip(prices.columns, beta.tolist()))
            result["calibration"] = filter1.calibrate_group(prices, result["beta"], prices.columns[leg])
    return result


def evaluate_groups(train: pd.DataFrame, groups: list[list[str]]) -> list[dict]:
    """evaluate_group en paralelo (un proceso por núcleo); el resultado no depende del orden ni del número de procesos."""
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        return list(pool.map(evaluate_group, [train[g] for g in groups], chunksize=4))


def run_funnel(train: pd.DataFrame, sector: str) -> tuple[pd.DataFrame, dict]:
    """Etapas 2 a 7 de un sector: tabla de todos sus subgrupos y detalles de los que llegaron a Johansen."""
    stage2 = correlation_stage(train)
    priority = engle_granger_stage(train, stage2)
    fallback = stage2_fallback_used(stage2)
    details = dict(zip(priority["subgroup"], evaluate_groups(train, [s.split("-") for s in priority["subgroup"]])))
    rows = []
    for _, row in stage2.merge(priority, on="subgroup", how="left").iterrows():
        base = {"sector": sector, **row.to_dict(), "stage2_fallback_used": fallback}
        if row["subgroup"] not in details:
            rows.append({**base, "stage_reached": 2, "decision": "descartado",
                         "reason": f"correlación media mínima por sub-ventana {row['corr_min_subwindow']:.3f} <= 0.70"})
            continue
        result = details[row["subgroup"]]
        rows.append({**base, **johansen_row(result["stage"]), "stability_share": result["stability_share"],
                     "stable_windows": result["stable_windows"]})
    table = pd.DataFrame(rows).sort_values(["stage_reached", "priority_rank"], ascending=[False, True])
    return table.reset_index(drop=True), details
