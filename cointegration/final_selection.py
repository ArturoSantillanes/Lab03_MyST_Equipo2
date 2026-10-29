"""Selección final del grupo entre todos los sectores: recorte de patas, siete requisitos y regla escrita de antemano.

Dueño: Milca
Lee: log-precios de train, la pertenencia de cada activo a su sector y los detalles del embudo de cada sector.
Escribe: (en memoria) la tabla de candidatos (va a artifacts/cointegration/candidatos_final.csv), el ganador y sus
detalles.
Reglas: SOLO train. El enunciado permite grupos de 2 a 8 activos; quitar patas sin peso económico no relaja ningún
criterio. Ningún umbral se toca. La regla está registrada en data/universe.json (preregistro v5) antes de ver datos.

Recorte: a cada subgrupo que cumple las etapas 4 a 6 se le quitan las patas cuya participación en el nocional,
|β_i| / Σ|β|, es menor que RELEVANCE_SHARE = 5% (con 6 activos el peso igual sería 16.7%; con el 80% de $1,000,000
el 5% son $40,000, monto en el que la comisión de 0.125% por lado anula el aporte de la pata), dejando al menos
MIN_GROUP_SIZE = 3 activos (los de mayor |β|). El recorte se evalúa completo y se vuelve a recortar si hace falta.

Requisitos (todos obligatorios para ser ELEGIBLE):
    R1 correlación media por pares de retornos > 0.7 en las 3 sub-ventanas
    R2 r̂ >= 1 (y < n) por la traza Y por el máximo eigenvalor al 95%
    R3 ADF del spread de Johansen p < 0.05
    R4 vida media del spread de Johansen entre 5 y 60 días
    R5 existe meseta de q (racha de >= 5 q válidos contiguos en la rejilla fina) con cero cambios de signo
    R6 ADF del spread de Kalman p < 0.05 en el q elegido (sin q válido no se cumple)
    R7 estabilidad >= 50% de las sub-ventanas
R3 y R4 se miden en la columna elegida; si no hay, en la columna de referencia: la de ADF más negativo dentro de r̂
(o entre todas si r̂ = 0), solo para poder contar requisitos.
Orden entre elegibles: mayor estabilidad, ADF de Johansen más negativo, menor burn-in (% de train), más activos.
Si ninguno es elegible no se baja ningún requisito: se ordena por número de requisitos cumplidos y luego igual.
"""
import numpy as np
import pandas as pd

from cointegration.correlation import MIN_MEAN_CORRELATION, rolling_pair_correlations, subwindow_mean_correlations
from cointegration.group_search import SIGN_CONVENTION, evaluate_groups
from cointegration.johansen import CRITICAL_VALUES_SOURCE, DETERMINISTIC
from cointegration.spread_checks import HALF_LIFE_RANGE, MAX_ADF_PVALUE, adf_test, half_life, normalize, spread_series

RELEVANCE_SHARE = 0.05
MIN_GROUP_SIZE = 3
MIN_STABILITY_SHARE = 0.50
REQUIREMENTS = {
    "R1": "correlación de retornos > 0.7 en las 3 sub-ventanas",
    "R2": "r̂ >= 1 por traza y por máximo eigenvalor al 95%",
    "R3": "ADF del spread de Johansen p < 0.05",
    "R4": "vida media de 5 a 60 días",
    "R5": "meseta de q con cero cambios de signo",
    "R6": "ADF del spread de Kalman p < 0.05",
    "R7": "estabilidad >= 50% de sub-ventanas",
}
SORT_KEYS = ["stability_share", "adf_stat", "burn_in_share", "n_assets"]
SORT_ASCENDING = [False, True, True, False]


def notional_shares(beta: dict) -> dict:
    """Participación de cada pata en el nocional: |β_i| / Σ|β|."""
    total = sum(abs(b) for b in beta.values())
    return {t: abs(b) / total for t, b in beta.items()}


def trimmed_group(beta: dict) -> list[str] | None:
    """Patas con participación >= RELEVANCE_SHARE (al menos MIN_GROUP_SIZE); None si no hay nada que quitar."""
    shares = notional_shares(beta)
    keep = [t for t in beta if shares[t] >= RELEVANCE_SHARE]
    if len(keep) == len(beta):
        return None
    if len(keep) < MIN_GROUP_SIZE:
        largest = sorted(beta, key=lambda t: shares[t], reverse=True)[:MIN_GROUP_SIZE]
        keep = [t for t in beta if t in largest]
    return keep if len(keep) < len(beta) else None


def reference_column(result: dict) -> pd.Series:
    """Columna elegida, o la de ADF más negativo dentro de r̂ (entre todas si r̂ = 0) para contar R3 y R4."""
    stage = result["stage"]
    columns = stage["columns"]
    if stage["chosen_column"] is not None:
        return columns.iloc[stage["chosen_column"]]
    pool = columns[columns["within_rank"]] if columns["within_rank"].any() else columns
    return pool.loc[pool["adf_stat"].idxmin()]


def candidate_row(sector: str, tickers: list[str], origin: str, result: dict, corr: list[float], n_train: int) -> dict:
    """Fila de la tabla de candidatos: estadísticos, filtro 1 y los siete requisitos."""
    j, n = result["stage"]["johansen"], len(tickers)
    column = reference_column(result)
    row = {"sector": sector, "group": "-".join(tickers), "n_assets": n, "origin": origin,
           "corr_min_subwindow": min(corr), "corr_subwindows": np.round(corr, 3).tolist(),
           "var_order_bic": j["var_order"], "r_hat_trace": j["rank_trace"], "r_hat_maxeig": j["rank_maxeig"],
           "trace_r0": j["trace_stats"][0], "trace_crit95_r0": j["trace_crit_95"][0], "trace_crit99_r0": j["trace_crit_99"][0],
           "maxeig_r0": j["maxeig_stats"][0], "maxeig_crit95_r0": j["maxeig_crit_95"][0],
           "eigenvalues": np.round(j["eigenvalues"], 5).tolist(), "chosen_column": result["stage"]["chosen_column"],
           "reference_column": int(column["column"]), "adf_stat": column["adf_stat"], "adf_pvalue": column["adf_pvalue"],
           "half_life_days": column["half_life_days"], "stability_share": result["stability_share"],
           "stable_windows": result["stable_windows"], "burn_in_share": np.inf}
    if result["beta"] is not None:
        row.update(beta={t: round(b, 4) for t, b in result["beta"].items()},
                   notional_share={t: round(s, 4) for t, s in notional_shares(result["beta"]).items()},
                   normalization_leg=max(result["beta"], key=lambda t: abs(result["beta"][t])))
    calibration = result["calibration"]
    if calibration is not None:
        row.update(q_grid_size=int((calibration["grid"]["resolution"] == "fina").sum()),
                   q_valid_count=calibration["q_valid_count"], q_longest_run=calibration["longest_run"],
                   q_plateau=calibration["plateau"], q_chosen=calibration["q"],
                   ultrafine_longest_run=calibration.get("ultrafine_longest_run"))
        if calibration["q"] is not None:
            m = calibration["metrics"]
            row.update(kalman_adf_pvalue=m["spread_adf_pvalue"], kalman_half_life_days=m["spread_half_life_days"],
                       sign_changes=int(m["sign_changes"]), burn_in_days=calibration["burn_in"],
                       burn_in_share=calibration["burn_in"] / n_train)
    checks = {
        "R1": min(corr) > MIN_MEAN_CORRELATION,
        "R2": 1 <= j["rank_trace"] < n and 1 <= j["rank_maxeig"] < n,
        "R3": column["adf_pvalue"] < MAX_ADF_PVALUE,
        "R4": HALF_LIFE_RANGE[0] <= column["half_life_days"] <= HALF_LIFE_RANGE[1],
        "R5": bool(calibration and calibration["plateau"]),
        "R6": bool(calibration and calibration["q"] is not None and row["kalman_adf_pvalue"] < MAX_ADF_PVALUE),
        "R7": result["stability_share"] >= MIN_STABILITY_SHARE,
    }
    row.update(checks)
    row.update(n_requirements=sum(checks.values()), eligible=all(checks.values()),
               fails="; ".join(f"{k} {REQUIREMENTS[k]}" for k, ok in checks.items() if not ok) or "ninguno")
    return row


def shortfalls(row: pd.Series) -> dict:
    """Para cada requisito que no cumple el grupo: valor logrado contra el objetivo."""
    text = {
        "R1": f"correlación mínima por sub-ventana {row['corr_min_subwindow']:.3f} (objetivo > 0.70)",
        "R2": f"r̂ traza {row['r_hat_trace']}, r̂ máx. eigenvalor {row['r_hat_maxeig']} (objetivo >= 1 en ambos)",
        "R3": f"ADF p = {row['adf_pvalue']:.4f} (objetivo < 0.05)",
        "R4": f"vida media {row['half_life_days']:.1f} días (objetivo 5 a 60)",
        "R5": f"racha válida más larga {row.get('q_longest_run', 0)} q contiguos (objetivo >= 5)",
        "R6": f"ADF del spread de Kalman p = {row.get('kalman_adf_pvalue', float('nan')):.4f} (objetivo < 0.05)",
        "R7": f"estabilidad {row['stability_share']:.0%} (objetivo >= 50%)",
    }
    return {k: text[k] for k in REQUIREMENTS if not row[k]}


def rank_candidates(table: pd.DataFrame) -> pd.DataFrame:
    """Orden de la regla: elegibles primero; dentro, número de requisitos y luego las claves de 4.2."""
    ordered = table.sort_values(["eligible", "n_requirements", *SORT_KEYS], ascending=[False, False, *SORT_ASCENDING])
    return ordered.assign(rank=range(1, len(ordered) + 1)).reset_index(drop=True)


def run_final_selection(train: pd.DataFrame, membership: dict, funnel_details: dict) -> tuple:
    """Evalúa todos los subgrupos que llegaron a Johansen y sus recortes; devuelve (tabla ordenada, ganador, detalles)."""
    sector_of = {t: s for s, tickers in membership.items() for t in tickers}
    pair_corr = {s: rolling_pair_correlations(train[tickers]) for s, tickers in membership.items()}
    evaluated = {group: ("embudo del sector", result) for group, result in funnel_details.items()}
    pending = evaluated
    while pending:
        trims = {}
        for group, (_, result) in pending.items():
            keep = trimmed_group(result["beta"]) if result["beta"] is not None else None
            if keep and "-".join(keep) not in evaluated and "-".join(keep) not in trims:
                trims["-".join(keep)] = f"recorte de {group} (patas < 5% del nocional)"
        results = evaluate_groups(train, [g.split("-") for g in trims])
        pending = {g: (origin, r) for (g, origin), r in zip(trims.items(), results)}
        evaluated = {**evaluated, **pending}
    rows = []
    for group, (origin, result) in evaluated.items():
        tickers = group.split("-")
        sector = sector_of[tickers[0]]
        corr = subwindow_mean_correlations(tuple(tickers), pair_corr[sector])
        rows.append(candidate_row(sector, tickers, origin, result, corr, len(train)))
    table = rank_candidates(pd.DataFrame(rows))
    winner = table.iloc[0]
    return table, winner, evaluated[winner["group"]][1], {g: r for g, (_, r) in evaluated.items()}



def describe_selection(result: dict, row: pd.Series, train: pd.DataFrame) -> dict:
    """Contenido de selected_group.json para el ganador."""
    stage, j = result["stage"], result["stage"]["johansen"]
    tickers = stage["subgroup"].split("-")
    beta, leg = normalize(j["eigenvectors"][:, stage["chosen_column"]])
    spread = spread_series(train[tickers], beta)
    stat, pvalue = adf_test(spread)
    return {
        "tickers": tickers,
        "sector": row["sector"],
        "origin": row["origin"],
        "period_used": "train",
        "price_transform": "log",
        "var_order": {"aic": stage["lags"]["aic"], "bic": stage["lags"]["bic"], "used": j["var_order"],
                      "used_rule": "BIC (parsimonia; fijado antes de ver resultados)"},
        "deterministic": DETERMINISTIC,
        "critical_values_source": CRITICAL_VALUES_SOURCE,
        "r_hat": j["rank_trace"],
        "r_hat_maxeig": j["rank_maxeig"],
        "eigenvalues": j["eigenvalues"],
        "trace_stats": j["trace_stats"], "trace_crit_95": j["trace_crit_95"], "trace_crit_99": j["trace_crit_99"],
        "maxeig_stats": j["maxeig_stats"], "maxeig_crit_95": j["maxeig_crit_95"], "maxeig_crit_99": j["maxeig_crit_99"],
        "beta_columns_raw": [dict(zip(tickers, col.tolist())) for col in j["eigenvectors"].T],
        "beta_columns_normalized": [dict(zip(tickers, normalize(col)[0].tolist())) for col in j["eigenvectors"].T],
        "column_table": stage["columns"].to_dict(orient="records"),
        "chosen_column": stage["chosen_column"],
        "beta": dict(zip(tickers, beta.tolist())),
        "abs_beta": dict(zip(tickers, np.abs(beta).tolist())),
        "notional_share": notional_shares(dict(zip(tickers, beta.tolist()))),
        "normalization_leg": tickers[leg],
        "sign_convention": SIGN_CONVENTION,
        "adf_stat": stat,
        "adf_pvalue": pvalue,
        "half_life_days": half_life(spread),
        "corr_subwindows": row["corr_subwindows"],
        "stability_share": float(row["stability_share"]),
        "stable_windows": row["stable_windows"],
        "requirements": {k: bool(row[k]) for k in REQUIREMENTS},
        "n_requirements": int(row["n_requirements"]),
        "eligible": bool(row["eligible"]),
        "shortfalls": shortfalls(row),
    }
