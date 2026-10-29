"""Embudo de selección (etapas 2 a 6) sobre los 84 subgrupos de 6 activos, solo con train.

Dueño: Milca
Lee: log-precios de train.
Escribe: (en memoria) la tabla de los 84 subgrupos y la descripción del grupo elegido.
Reglas: SOLO train. Contar todo lo que se prueba.

Respaldo de la etapa 2 (escrito DESPUÉS de ver que ningún subgrupo pasa la heurística de correlación, y
contado como respaldo): como la correlación es una heurística y no el criterio definitivo, si ningún subgrupo
la pasa, la etapa 3 prioriza los 84 y el criterio definitivo sigue siendo Johansen (etapas 4 a 6).

Regla secuencial, fijada antes de ver resultados: se toma el subgrupo con mayor prioridad de Engle-Granger,
se le aplica Johansen (etapa 4) y se validan sus columnas (etapas 5 y 6). Si no cumple, se pasa al
siguiente de la lista; se detiene en el primero que cumple. r̂ = 0 o r̂ = n cuentan como "no cumple".
"""
import itertools

import pandas as pd

from cointegration.correlation import (mean_level_correlation, passes_correlation, rolling_pair_correlations,
                                       subwindow_mean_correlations)
from cointegration.engle_granger import group_pair_stats, priority_order
from cointegration.johansen import CRITICAL_VALUES_SOURCE, DETERMINISTIC, johansen_test, rank_alarm, var_lag_orders
from cointegration.spread_checks import (HALF_LIFE_RANGE, adf_test, choose_column, column_table, half_life, normalize,
                                         spread_series)

GROUP_SIZE = 6
SIGN_CONVENTION = ("largo el spread = comprar cada activo con β_i > 0 y vender cada activo con β_i < 0, en proporción "
                   "a |β_i|; corto el spread = todos los signos al revés")


def correlation_stage(train: pd.DataFrame) -> pd.DataFrame:
    """Etapa 2 para los C(9,6) subgrupos: correlación media por sub-ventana y si pasa."""
    pair_corr = rolling_pair_correlations(train)
    rows = []
    for group in itertools.combinations(train.columns, GROUP_SIZE):
        corr = subwindow_mean_correlations(group, pair_corr)
        rows.append({"subgroup": "-".join(group), **{f"corr_subwindow_{i + 1}": c for i, c in enumerate(corr)},
                     "corr_min_subwindow": min(corr), "stage2_pass": passes_correlation(corr),
                     "info_corr_log_levels": mean_level_correlation(group, train)})
    return pd.DataFrame(rows)


def stage2_fallback_used(stage2: pd.DataFrame) -> bool:
    """True si ningún subgrupo pasó la heurística de correlación y la etapa 3 debe priorizar los 84."""
    return not stage2["stage2_pass"].any()


def engle_granger_stage(train: pd.DataFrame, stage2: pd.DataFrame) -> pd.DataFrame:
    """Etapa 3: estadísticos por pares y orden de prioridad (de los que pasaron la etapa 2, o de los 84 si hay respaldo)."""
    candidates = stage2 if stage2_fallback_used(stage2) else stage2[stage2["stage2_pass"]]
    stats = pd.DataFrame([{"subgroup": s, **group_pair_stats(train, tuple(s.split("-")))} for s in candidates["subgroup"]])
    return priority_order(stats)


def johansen_stage(train: pd.DataFrame, subgroup: str) -> dict:
    """Etapas 4, 5 y 6 para un subgrupo: Johansen, tabla de columnas, columna elegida y validación."""
    prices = train[subgroup.split("-")]
    lags = var_lag_orders(prices)
    johansen = johansen_test(prices, lags["bic"])
    rank = johansen["rank_trace"]
    alarm = rank_alarm(rank, prices.shape[1])
    columns = column_table(prices, johansen["eigenvectors"], rank)
    chosen = None if alarm else choose_column(columns)
    return {"subgroup": subgroup, "lags": lags, "johansen": johansen, "alarm": alarm, "columns": columns,
            "chosen_column": chosen}


def johansen_row(result: dict) -> dict:
    """Columnas de la tabla de selección para un subgrupo que llegó a Johansen."""
    j = result["johansen"]
    row = {"var_order_aic": result["lags"]["aic"], "var_order_bic": result["lags"]["bic"],
           "rank_trace": j["rank_trace"], "rank_maxeig": j["rank_maxeig"],
           "trace_r0": j["trace_stats"][0], "trace_crit95_r0": j["trace_crit_95"][0],
           "maxeig_r0": j["maxeig_stats"][0], "maxeig_crit95_r0": j["maxeig_crit_95"][0]}
    if result["chosen_column"] is not None:
        best = result["columns"].iloc[result["chosen_column"]]
        row.update(chosen_column=int(best["column"]), adf_stat=best["adf_stat"], adf_pvalue=best["adf_pvalue"],
                   half_life_days=best["half_life_days"])
    return row


def sequential_johansen(train: pd.DataFrame, priority: pd.DataFrame) -> tuple[dict, dict]:
    """Recorre la lista de prioridad hasta el primer subgrupo que cumple las etapas 4 a 6."""
    rows, chosen = {}, None
    for subgroup in priority["subgroup"]:
        result = johansen_stage(train, subgroup)
        row = johansen_row(result)
        if result["alarm"]:
            row.update(stage_reached=4, decision="rechazado en Johansen", reason=result["alarm"])
        elif result["chosen_column"] is None:
            row.update(stage_reached=6, decision="rechazado en validación del spread",
                       reason=f"ninguna columna dentro de r̂ con ADF p < 0.05 y vida media en {HALF_LIFE_RANGE} días")
        else:
            row.update(stage_reached=6, decision="elegido", reason="primer subgrupo de la lista que cumple etapas 4 a 6")
            chosen = result
        rows[subgroup] = row
        if chosen:
            break
    return rows, chosen


def selection_table(stage2: pd.DataFrame, priority: pd.DataFrame, johansen_rows: dict) -> pd.DataFrame:
    """Une las etapas en una fila por subgrupo con etapa alcanzada, estadísticos, decisión y razón."""
    table = stage2.merge(priority, on="subgroup", how="left")
    defaults = []
    fallback = stage2_fallback_used(stage2)
    for _, row in table.iterrows():
        if not row["stage2_pass"] and not fallback:
            defaults.append({"stage_reached": 2, "decision": "descartado",
                             "reason": f"correlación media mínima por sub-ventana {row['corr_min_subwindow']:.3f} <= 0.70"})
        else:
            note = " (no pasó la heurística de correlación; priorizado por el respaldo)" if fallback else ""
            defaults.append({"stage_reached": 3, "decision": "priorizado, no llegó a Johansen",
                             "reason": "ya se había encontrado un subgrupo que cumple con mayor prioridad" + note})
    table = pd.concat([table.reset_index(drop=True), pd.DataFrame(defaults)], axis=1)
    for subgroup, row in johansen_rows.items():
        for key, value in row.items():
            table.loc[table["subgroup"] == subgroup, key] = value
    return table


def describe_selection(chosen: dict, train: pd.DataFrame, n_evaluated: int, n_johansen: int, fallback: bool) -> dict:
    """Contenido de selected_group.json."""
    tickers = chosen["subgroup"].split("-")
    j = chosen["johansen"]
    raw = j["eigenvectors"][:, chosen["chosen_column"]]
    beta, leg = normalize(raw)
    spread = spread_series(train[tickers], beta)
    stat, pvalue = adf_test(spread)
    return {
        "tickers": tickers,
        "period_used": "train",
        "price_transform": "log",
        "n_subgroups_evaluated": n_evaluated,
        "n_subgroups_reached_johansen": n_johansen,
        "stage2_fallback_used": fallback,
        "fallback_count": int(fallback),
        "var_order": {"aic": chosen["lags"]["aic"], "bic": chosen["lags"]["bic"], "used": j["var_order"],
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
        "column_table": chosen["columns"].to_dict(orient="records"),
        "chosen_column": chosen["chosen_column"],
        "beta": dict(zip(tickers, beta.tolist())),
        "normalization_leg": tickers[leg],
        "sign_convention": SIGN_CONVENTION,
        "adf_stat": stat,
        "adf_pvalue": pvalue,
        "half_life_days": half_life(spread),
    }


def run_funnel(train: pd.DataFrame) -> tuple[pd.DataFrame, dict | None]:
    """Etapas 2 a 6 completas: tabla de los 84 subgrupos y grupo elegido (o None si ninguno cumple)."""
    stage2 = correlation_stage(train)
    priority = engle_granger_stage(train, stage2)
    johansen_rows, chosen = sequential_johansen(train, priority)
    table = selection_table(stage2, priority, johansen_rows)
    fallback = stage2_fallback_used(stage2)
    selection = describe_selection(chosen, train, len(table), len(johansen_rows), fallback) if chosen else None
    return table, selection
