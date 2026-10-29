"""Selección final del grupo: diagnóstico de patas, candidatos y regla de selección escrita de antemano (solo train).

Dueño: Milca
Lee: log-precios de train, el grupo que eligió el embudo y la tabla de los 84 subgrupos.
Escribe: (en memoria) la tabla de candidatos (va a artifacts/cointegration/candidatos_final.csv) y el ganador.
Reglas: SOLO train. El enunciado permite grupos de 2 a 8 activos; quitar patas sin peso económico no relaja ningún
criterio. El umbral de las pruebas no se toca.

Peso económicamente relevante: una pata es relevante si su participación en el nocional, |β_i| / Σ|β|, es de al
menos RELEVANCE_SHARE = 5%. Con 6 activos el peso igual sería 16.7%; menos de un tercio de eso no aporta exposición, y
con el 80% de $1,000,000 el 5% son $40,000, monto en el que la comisión de 0.125% por lado anula el aporte de la pata.
Candidatos: (a) el grupo del embudo; (b) quitarle 1, 2 y 3 de sus patas de menor |β| (mínimo MIN_GROUP_SIZE activos);
(c) todos los subgrupos que llegaron a Johansen y cumplieron las etapas 5 y 6 en el embudo.
Regla de selección (fijada antes de ver la tabla):
    elegible si r̂ >= 1 con la traza Y con el máximo eigenvalor al 95%, ADF del spread p < 0.05, vida media entre
    5 y 60 días, y existe un q que cumple (a)-(d) del filtro 1 (en particular CERO cambios de signo);
    entre los elegibles: mayor fracción de sub-ventanas estables; desempate: ADF más negativo; luego más activos.
    Si ninguno es elegible no se relaja nada.
"""
import numpy as np
import pandas as pd

from cointegration.group_search import SIGN_CONVENTION, johansen_stage
from cointegration.johansen import DETERMINISTIC
from cointegration.spread_checks import HALF_LIFE_RANGE, MAX_ADF_PVALUE, normalize
from cointegration.stability import stability_share, stability_windows
import kalman.filter1_weights as filter1  # módulo completo: evita la importación circular cointegration <-> kalman

RELEVANCE_SHARE = 0.05
MIN_GROUP_SIZE = 3


def notional_shares(beta: dict) -> dict:
    """Participación de cada pata en el nocional: |β_i| / Σ|β|."""
    total = sum(abs(b) for b in beta.values())
    return {t: abs(b) / total for t, b in beta.items()}


def candidate_groups(funnel_tickers: list[str], funnel_beta: dict, funnel_table: pd.DataFrame) -> list[tuple]:
    """Lista de (tickers, origen) sin repetidos: el grupo del embudo, sus recortes por |β| y los que cumplieron 5-6."""
    by_weight = sorted(funnel_tickers, key=lambda t: abs(funnel_beta[t]))
    candidates = [(funnel_tickers, "grupo del embudo")]
    for k in range(1, len(funnel_tickers) - MIN_GROUP_SIZE + 1):
        dropped = set(by_weight[:k])
        candidates.append(([t for t in funnel_tickers if t not in dropped], f"quitar {k} pata(s) de menor |β|: {sorted(dropped)}"))
    passed = funnel_table.loc[funnel_table["decision"] == "elegido", "subgroup"]
    candidates += [(s.split("-"), "cumplió etapas 5 y 6 en el embudo") for s in passed]
    unique, seen = [], set()
    for tickers, origin in candidates:
        if tuple(tickers) not in seen:
            seen.add(tuple(tickers))
            unique.append((tickers, origin))
    return unique


def evaluate_candidate(train: pd.DataFrame, tickers: list[str], origin: str) -> tuple[dict, dict]:
    """Proceso completo en train para un candidato; devuelve (fila de la tabla, detalles para el ganador)."""
    stage = {"subgroup": "-".join(tickers), **johansen_stage(train, "-".join(tickers))}
    j, columns = stage["johansen"], stage["columns"]
    row = {"group": stage["subgroup"], "n_assets": len(tickers), "origin": origin,
           "var_order_aic": stage["lags"]["aic"], "var_order_bic": stage["lags"]["bic"], "deterministic": DETERMINISTIC,
           "r_hat_trace": j["rank_trace"], "r_hat_maxeig": j["rank_maxeig"], "tests_agree": j["rank_trace"] == j["rank_maxeig"],
           "eigenvalues": np.round(j["eigenvalues"], 5).tolist(), "trace_stats": np.round(j["trace_stats"], 2).tolist(),
           "trace_crit_95": j["trace_crit_95"], "maxeig_stats": np.round(j["maxeig_stats"], 2).tolist(),
           "maxeig_crit_95": j["maxeig_crit_95"], "columns_adf": np.round(columns["adf_stat"], 3).tolist(),
           "columns_half_life": np.round(columns["half_life_days"], 1).tolist(), "chosen_column": stage["chosen_column"]}
    windows = stability_windows(train[tickers], j["var_order"])
    row.update(stability_share=stability_share(windows), stable_windows=f"{int(windows['stable'].sum())}/{len(windows)}")
    if stage["chosen_column"] is None:
        return {**row, **eligibility(row)}, {"stage": stage, "windows": windows}
    beta, leg = normalize(j["eigenvectors"][:, stage["chosen_column"]])
    beta = dict(zip(tickers, beta.tolist()))
    best = columns.iloc[stage["chosen_column"]]
    calibration = filter1.calibrate_group(train[tickers], beta, tickers[leg])
    row.update(adf_stat=best["adf_stat"], adf_pvalue=best["adf_pvalue"], half_life_days=best["half_life_days"],
               normalization_leg=tickers[leg], sign_convention=SIGN_CONVENTION,
               abs_beta={t: round(abs(b), 4) for t, b in beta.items()},
               notional_share={t: round(s, 4) for t, s in notional_shares(beta).items()},
               q_grid_size=len(calibration["grid"]), q_passing=int(calibration["grid"]["ok_all"].sum()), q_chosen=calibration["q"],
               min_sign_changes_in_grid=int(calibration["grid"]["sign_changes"].min()))
    if calibration["q"] is not None:
        row.update(sign_changes_total=sum(calibration["sign_changes_by_asset"].values()),
                   sign_changes_by_asset=calibration["sign_changes_by_asset"], burn_in_days=calibration["burn_in"],
                   posvar_floor={t: float(f"{v:.3g}") for t, v in calibration["posvar_floor"].items()})
    return {**row, **eligibility(row)}, {"stage": stage, "windows": windows, "beta": beta}


def eligibility(row: dict) -> dict:
    """Aplica el filtro obligatorio de la regla y explica qué falla."""
    checks = {
        "r̂ >= 1 con traza y máximo eigenvalor": row["r_hat_trace"] >= 1 and row["r_hat_maxeig"] >= 1,
        "ADF p < 0.05": row.get("adf_pvalue", 1.0) < MAX_ADF_PVALUE,
        "vida media 5-60 días": HALF_LIFE_RANGE[0] <= row.get("half_life_days", np.inf) <= HALF_LIFE_RANGE[1],
        "q con cero cambios de signo que cumple (a)-(d)": row.get("q_chosen") is not None,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {"eligible": not failed, "fails": "; ".join(failed) if failed else "ninguno"}


def select_winner(table: pd.DataFrame) -> pd.Series | None:
    """Ganador entre los elegibles: mayor estabilidad, luego ADF más negativo, luego más activos."""
    eligible = table[table["eligible"]]
    if eligible.empty:
        return None
    return eligible.sort_values(["stability_share", "adf_stat", "n_assets"], ascending=[False, True, False]).iloc[0]


def run_final_selection(train: pd.DataFrame, funnel_selection: dict, funnel_table: pd.DataFrame) -> tuple:
    """Evalúa todos los candidatos y devuelve (tabla, fila ganadora o None, detalles del ganador)."""
    evaluated = [evaluate_candidate(train, tickers, origin)
                 for tickers, origin in candidate_groups(funnel_selection["tickers"], funnel_selection["beta"], funnel_table)]
    table = pd.DataFrame([row for row, _ in evaluated])
    winner = select_winner(table)
    details = None if winner is None else next(d for row, d in evaluated if row["group"] == winner["group"])
    return table, winner, details
