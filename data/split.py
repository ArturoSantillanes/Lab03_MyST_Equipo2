"""Partición cronológica 60/20/20 en train / test / valid, sin traslape y sin look-ahead.

Dueño: Milca
Lee: data/processed/prices.csv
Escribe: data/processed/splits.json
Reglas: Cronológico, sin barajar. La validación se usa UNA sola vez, al final.

Corte por número de observaciones: train = primeras floor(0.6·n), test = hasta floor(0.8·n), valid = el resto.
"""
import pandas as pd

TRAIN_SHARE, TEST_SHARE = 0.60, 0.20
PERIODS = ("train", "test", "valid")


def compute_splits(dates: pd.DatetimeIndex) -> dict:
    """Inicio, fin y número de observaciones de cada tramo."""
    n = len(dates)
    cuts = {"train": (0, int(n * TRAIN_SHARE)), "test": (int(n * TRAIN_SHARE), int(n * (TRAIN_SHARE + TEST_SHARE))),
            "valid": (int(n * (TRAIN_SHARE + TEST_SHARE)), n)}
    splits = {name: {"start": dates[lo].strftime("%Y-%m-%d"), "end": dates[hi - 1].strftime("%Y-%m-%d"), "n_obs": hi - lo}
              for name, (lo, hi) in cuts.items()}
    splits["n_total"] = n
    return splits


def assign_period(dates: pd.DatetimeIndex, splits: dict) -> pd.Series:
    """Etiqueta cada fecha como train, test o valid."""
    labels = pd.Series(pd.NA, index=dates, name="period", dtype="object")
    for name in PERIODS:
        labels[(dates >= pd.Timestamp(splits[name]["start"])) & (dates <= pd.Timestamp(splits[name]["end"]))] = name
    return labels


def slice_period(frame: pd.DataFrame, splits: dict, period: str) -> pd.DataFrame:
    """Filas de `frame` dentro del tramo indicado."""
    return frame.loc[splits[period]["start"]:splits[period]["end"]]
