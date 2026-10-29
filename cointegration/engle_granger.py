"""Etapa 3 del embudo: pre-filtro de Engle-Granger por pares. PRIORIZA, NO ELIMINA.

Dueño: Milca
Lee: log-precios de train del sector (los pares se calculan una vez y se reutilizan en cada subgrupo).
Escribe: (en memoria) número de pares cointegrados, p-value medio y orden de prioridad.
Reglas: SOLO train. Ningún subgrupo se elimina aquí: todos quedan en la lista ordenada.

Para cada uno de los 15 pares de un subgrupo de 6: regresión OLS del primer activo (orden de
data/universe.json) contra el segundo, con constante, y ADF sobre los residuales (statsmodels.coint, valores
críticos de MacKinnon). Un par "pasa" si p < MAX_PVALUE. Prioridad: más pares que pasan primero; desempate
por menor p-value medio.
"""
import itertools

import pandas as pd
from statsmodels.tsa.stattools import coint

MAX_PVALUE = 0.05


def pair_pvalue(log_prices: pd.DataFrame, a: str, b: str) -> float:
    """p-value de Engle-Granger del par (a dependiente, b regresor)."""
    return float(coint(log_prices[a], log_prices[b], trend="c")[1])


def pair_pvalues(log_prices: pd.DataFrame) -> dict:
    """p-value de Engle-Granger de cada par del sector (orden de columnas de data/universe.json)."""
    return {(a, b): pair_pvalue(log_prices, a, b) for a, b in itertools.combinations(log_prices.columns, 2)}


def group_pair_stats(pvalues_by_pair: dict, group: tuple) -> dict:
    """Pares que pasan p < MAX_PVALUE y p-value medio del subgrupo."""
    pvalues = [pvalues_by_pair[pair] for pair in itertools.combinations(group, 2)]
    return {"eg_pairs_pass": sum(p < MAX_PVALUE for p in pvalues), "eg_pairs_total": len(pvalues),
            "eg_mean_pvalue": sum(pvalues) / len(pvalues)}


def priority_order(stats: pd.DataFrame) -> pd.DataFrame:
    """Ordena de mayor a menor prioridad (más pares que pasan; desempate: menor p medio) y numera."""
    ordered = stats.sort_values(["eg_pairs_pass", "eg_mean_pvalue"], ascending=[False, True])
    return ordered.assign(priority_rank=range(1, len(ordered) + 1))
