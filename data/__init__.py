"""Etapa 'data' del pipeline: descarga (o lee congelado), valida, limpia y parte los datos de todos los sectores.

Dueño: Milca
Lee: data/universe.json, Yahoo Finance (solo la primera vez) o data/raw/<sector>.csv
Escribe: data/processed/prices.csv (todos los candidatos de todos los sectores, precios ajustados en USD, con
         columna period), data/processed/sectors.json (tickers que quedan en cada sector y los excluidos),
         data/processed/cleaning_report.json, data/processed/splits.json
Reglas: Datos diarios de 10 a 15 años; partición 60/20/20 cronológica sin look-ahead. No usa aleatoriedad
(la semilla 42 está fijada en main.py). El resto del bloque trabaja con el logaritmo natural de estos precios.
"""
import json
from pathlib import Path

import pandas as pd

from cointegration.universe import load_universe, sector_tickers
from data.clean import clean_prices, incomplete_tickers
from data.download import load_or_download
from data.split import assign_period, compute_splits

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
PRICES_PATH = PROCESSED_DIR / "prices.csv"
SECTORS_PATH = PROCESSED_DIR / "sectors.json"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"


def load_sectors(universe: dict) -> tuple[pd.DataFrame, dict]:
    """Lee cada sector, excluye los tickers con historia incompleta y une todo por fecha."""
    window = universe["window"]
    frames, membership = [], {}
    for sector in universe["sectors"]:
        raw = load_or_download(sector_tickers(sector), sector["raw_file"], window["start"], window["end"])
        excluded = incomplete_tickers(raw)
        kept = [t for t in raw.columns if t not in excluded]
        membership[sector["id"]] = {"tickers": kept, "excluded": excluded}
        frames.append(raw[kept])
    return pd.concat(frames, axis=1), membership


def run() -> None:
    """Obtiene los precios de todos los sectores, los limpia, calcula la partición y guarda los archivos."""
    raw, membership = load_sectors(load_universe())
    prices, report = clean_prices(raw)
    report["excluded_tickers"] = {s: m["excluded"] for s, m in membership.items() if m["excluded"]}
    splits = compute_splits(prices.index)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.assign(period=assign_period(prices.index, splits)).to_csv(PRICES_PATH, date_format="%Y-%m-%d")
    SECTORS_PATH.write_text(json.dumps(membership, indent=2, ensure_ascii=False))
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print(f"  {len(membership)} sectores, {len(prices.columns)} activos, {len(prices)} días ({report['start']} a "
          f"{report['end']}); excluidos {report['excluded_tickers'] or 'ninguno'}; duplicados {report['duplicate_dates']}, "
          f"desalineadas {report['misaligned_dates']}, saltos marcados {report['anomalous_jumps_count']} "
          f"(posibles errores {report['possible_data_errors_count']})")
    for name in ("train", "test", "valid"):
        print(f"  {name}: {splits[name]['start']} a {splits[name]['end']} ({splits[name]['n_obs']} días)")
