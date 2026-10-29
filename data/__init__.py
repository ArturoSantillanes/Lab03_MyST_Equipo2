"""Etapa 'data' del pipeline: descarga (o lee congelado), valida, limpia y parte los datos.

Dueño: Milca
Lee: data/universe.json, Yahoo Finance (solo la primera vez) o data/raw/prices_raw.csv
Escribe: data/processed/prices.csv (los 9 activos del universo, precios ajustados en USD, con columna period),
         data/processed/cleaning_report.json, data/processed/splits.json
Reglas: Datos diarios de 10 a 15 años; partición 60/20/20 cronológica sin look-ahead. No usa aleatoriedad
(la semilla 42 está fijada en main.py). El resto del bloque trabaja con el logaritmo natural de estos precios.
"""
import json
from pathlib import Path

from cointegration.universe import candidate_tickers, load_universe
from data.clean import clean_prices
from data.download import load_or_download
from data.split import assign_period, compute_splits

PROCESSED_DIR = Path(__file__).resolve().parent / "processed"
PRICES_PATH = PROCESSED_DIR / "prices.csv"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
REPORT_PATH = PROCESSED_DIR / "cleaning_report.json"


def run() -> None:
    """Obtiene los precios del universo, los limpia, calcula la partición y guarda los archivos."""
    universe = load_universe()
    window = universe["window"]
    prices, report = clean_prices(load_or_download(candidate_tickers(universe), window["start"], window["end"]))
    splits = compute_splits(prices.index)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    prices.assign(period=assign_period(prices.index, splits)).to_csv(PRICES_PATH, date_format="%Y-%m-%d")
    SPLITS_PATH.write_text(json.dumps(splits, indent=2))
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    print(f"  {len(prices.columns)} activos, {len(prices)} días ({report['start']} a {report['end']}); "
          f"duplicados {report['duplicate_dates']}, desalineadas {report['misaligned_dates']}, "
          f"saltos marcados {report['anomalous_jumps_count']} (posibles errores {report['possible_data_errors_count']})")
    for name in ("train", "test", "valid"):
        print(f"  {name}: {splits[name]['start']} a {splits[name]['end']} ({splits[name]['n_obs']} días)")
