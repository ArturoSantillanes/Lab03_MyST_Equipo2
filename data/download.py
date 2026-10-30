"""Descarga los precios diarios ajustados de cada sector candidato y los congela (un archivo por sector).

Dueño: Milca
Lee: Yahoo Finance (librería yfinance), solo si no existe la descarga congelada del sector.
Escribe: data/raw/<sector>.csv y data/raw/<sector>_meta.json
Reglas: Ventana de 10 a 15 años (data/universe.json). Registrar fecha de descarga y fuente.

Se usa el cierre ajustado por dividendos y splits (auto_adjust=True). Cada sector se descarga una sola vez;
después el pipeline lee el archivo congelado y no necesita internet ni yfinance.
"""
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


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


def freeze(prices: pd.DataFrame, raw_path: Path, start: str, end: str) -> None:
    """Guarda la descarga y su registro (fuente, fecha, ventana, versión de yfinance)."""
    import yfinance as yf

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_csv(raw_path, date_format="%Y-%m-%d")
    raw_path.with_name(f"{raw_path.stem}_meta.json").write_text(json.dumps({
        "source": "Yahoo Finance (yfinance), cierre ajustado por dividendos y splits",
        "yfinance_version": yf.__version__,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": start,
        "end": end,
        "tickers": list(prices.columns),
    }, indent=2))


def load_or_download(tickers: list[str], raw_file: str, start: str, end: str) -> pd.DataFrame:
    """Lee la descarga congelada del sector; si no existe, descarga y congela."""
    raw_path = ROOT / raw_file
    if not raw_path.exists():
        freeze(download_prices(tickers, start, end), raw_path, start, end)
    return pd.read_csv(raw_path, index_col="date", parse_dates=True)
