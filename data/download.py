"""Descarga los precios diarios ajustados del universo de candidatos y los congela.

Dueño: Milca
Lee: Yahoo Finance (librería yfinance), solo si no existe la descarga congelada.
Escribe: data/raw/prices_raw.csv y data/raw/download_meta.json
Reglas: Ventana de 10 a 15 años (data/universe.json). Registrar fecha de descarga y fuente.

Se usa el cierre ajustado por dividendos y splits (auto_adjust=True). La descarga se hace
una sola vez; después el pipeline lee el archivo congelado y no necesita internet ni yfinance.
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent / "raw"
RAW_PRICES_PATH = RAW_DIR / "prices_raw.csv"
RAW_META_PATH = RAW_DIR / "download_meta.json"


def download_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Descarga cierres ajustados; una columna por ticker. `end` es inclusivo."""
    import yfinance as yf  # solo se necesita si no hay descarga congelada

    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download([t.upper() for t in tickers], start=start, end=end_exclusive,
                          auto_adjust=True, progress=False)
    close = raw["Close"].rename(columns=str.lower).reindex(columns=tickers)
    close.index = pd.DatetimeIndex(close.index).tz_localize(None)
    close.index.name = "date"
    return close


def freeze(prices: pd.DataFrame, start: str, end: str) -> None:
    """Guarda la descarga y su registro (fuente, fecha, ventana, versión de yfinance)."""
    import yfinance as yf

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    prices.to_csv(RAW_PRICES_PATH, date_format="%Y-%m-%d")
    RAW_META_PATH.write_text(json.dumps({
        "source": "Yahoo Finance (yfinance), cierre ajustado por dividendos y splits",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start,
        "end": end,
        "tickers": list(prices.columns),
    }, indent=2))


def load_or_download(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Lee la descarga congelada; si no existe, descarga y congela."""
    if not RAW_PRICES_PATH.exists():
        freeze(download_prices(tickers, start, end), start, end)
    return pd.read_csv(RAW_PRICES_PATH, index_col="date", parse_dates=True)
