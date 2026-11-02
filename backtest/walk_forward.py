"""Esquema walk-forward para evaluar en test y valid sin reajustar con ellos.

Dueño: Paula
Lee: Precios, señales y parámetros fijados en train.
Escribe: Resultados por periodo.
Reglas: La validación se usa UNA sola vez, al final.

Versión implementada (supuesto explícito; el enunciado nombra "walk-forward" sin describirlo, queda como pregunta para
el profesor): un solo paso de ANCLA FIJA. Se calibra en train, se congelan los parámetros y se evalúa en secuencia
train → test → valid, cada tramo como una corrida independiente que empieza plana con $1,000,000 y cierra al final
del tramo. No hay re-estimación en ventanas rodantes: los filtros de Kalman ya se adaptan día a día con parámetros de
train, y re-calibrar umbrales con test usaría test para decidir.
La validación se evalúa UNA vez por corrida del pipeline: ValidationGuard cuenta las evaluaciones y truena si se pide
una segunda.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import BacktestResult, run_backtest
from backtest.io import read_json
from strategy import generate_positions

ROOT = Path(__file__).resolve().parent.parent
PERIODS = ("train", "test", "valid")


@dataclass
class Inputs:
    """Insumos del backtest ya alineados por fecha: precios en niveles del grupo, β̂_t, z operable y periodo."""
    prices: pd.DataFrame
    beta: pd.DataFrame
    z: pd.Series
    z_frame: pd.DataFrame
    period: pd.Series
    splits: dict
    selection: dict
    params_default: dict


def load_inputs(root: Path = ROOT) -> Inputs:
    """Lee prices.csv (solo las 6 columnas del grupo), beta_t.csv, z_t.csv, splits.json y selected_group.json."""
    selection = read_json(root / "artifacts" / "cointegration" / "selected_group.json")
    tickers = selection["tickers"]
    prices = pd.read_csv(root / "data" / "processed" / "prices.csv", index_col="date", parse_dates=True)
    beta = pd.read_csv(root / "artifacts" / "kalman" / "beta_t.csv", index_col="date", parse_dates=True)
    z_frame = pd.read_csv(root / "artifacts" / "kalman" / "z_t.csv", index_col="date", parse_dates=True)
    z = z_frame["z"].where(~z_frame["burn_in"].astype(bool))
    return Inputs(prices=prices[tickers], beta=beta[tickers], z=z, z_frame=z_frame, period=z_frame["period"],
                  splits=read_json(root / "data" / "processed" / "splits.json"), selection=selection,
                  params_default=read_json(root / "artifacts" / "strategy" / "params_default.json"))


def period_z(inputs: Inputs, period: str) -> pd.Series:
    """z operable del tramo; en train empieza el primer día sin burn-in (2015-05-21)."""
    z = inputs.z[inputs.period == period]
    first = z.first_valid_index()
    return z.loc[first:]


@dataclass
class ValidationGuard:
    """Cuenta cuántas veces se evaluó valid y prohíbe una segunda evaluación."""
    count: int = 0
    log: list = field(default_factory=list)

    def touch(self, label: str) -> None:
        """Registra la evaluación de valid; truena si ya se había hecho."""
        if self.count >= 1:
            raise RuntimeError("La validación ya se evaluó una vez en esta corrida; no se puede volver a evaluar.")
        self.count += 1
        self.log.append(label)


def evaluate_period(inputs: Inputs, period: str, params: dict, band: float) -> tuple[pd.DataFrame, BacktestResult]:
    """Posiciones y backtest de un tramo con parámetros congelados."""
    positions = generate_positions(period_z(inputs, period), params)
    return positions, run_backtest(inputs.prices, inputs.beta, positions, band=band)


def walk_forward(inputs: Inputs, params: dict, bands: dict, guard: ValidationGuard) -> dict:
    """Evalúa train → test → valid en ese orden con params congelados, para cada política de `bands` (nombre → banda).

    Todas las políticas se evalúan en la MISMA y única pasada por valid (una evaluación registrada en guard).
    Devuelve {política: {tramo: (posiciones, resultado)}}.
    """
    out = {name: {} for name in bands}
    for period in PERIODS:
        if period == "valid":
            guard.touch(f"walk_forward con params congelados {params} y políticas {list(bands)}")
        for name, band in bands.items():
            out[name][period] = evaluate_period(inputs, period, params, band)
    return out


def stack(results: dict, attribute: str) -> pd.DataFrame:
    """Une los tramos de una política en un solo DataFrame con columna period."""
    frames = []
    for period, (_, result) in results.items():
        frame = getattr(result, attribute).copy()
        frame.insert(0, "period", period)
        frames.append(frame)
    return pd.concat(frames)


def exit_reason_counts(trades: pd.DataFrame) -> dict:
    """Operaciones por motivo de salida."""
    reasons = ["senal", "stop", "tenencia_maxima", "fin_de_periodo"]
    counts = trades["exit_reason"].value_counts()
    return {reason: int(counts.get(reason, 0)) for reason in reasons}


def stop_diagnostics(inputs: Inputs, results: dict, params: dict) -> dict:
    """Verificación del hallazgo de Arturo: ¿protege el stop sobre |z| cuando el spread se dispara?

    Por tramo: motivos de salida con los parámetros finales y con los de arranque (solo señales, sin P&L), máximo |z| y
    peor movimiento adverso del spread dentro de cada operación, y cuántos días el spread se alejó de su media más de
    z_stop veces la σ̂ mediana de train contra cuántos días |z| pasó de z_stop.
    """
    frame = inputs.z_frame
    deviation = frame["spread"] - frame["mu_prev"]
    operable = ~frame["burn_in"].astype(bool)
    sigma_train = float(frame.loc[operable & (frame["period"] == "train"), "sigma_prev"].median())
    out = {"sigma_prev_median_train": sigma_train, "by_period": {}}
    for period, (_, result) in results.items():
        mask = operable & (frame["period"] == period)
        default_positions = generate_positions(period_z(inputs, period), inputs.params_default)
        default_reasons = default_positions["exit_reason"].replace("", np.nan).dropna().value_counts().to_dict()
        trades = result.trades
        worst_z, worst_move = [], []
        for trade in trades.itertuples():
            window = frame.loc[trade.entry_date:trade.exit_date]
            sign = 1 if trade.direction == "largo" else -1
            worst_z.append(float(window["z"].abs().max()))
            worst_move.append(float((-sign * (window["spread"] - window["spread"].iloc[0])).max()))
        stop = params["z_stop"]
        out["by_period"][period] = {
            "exit_reasons_final_params": exit_reason_counts(trades),
            "exit_reasons_default_params_signals_only": {k: int(v) for k, v in default_reasons.items()},
            "max_abs_z_inside_trades": float(max(worst_z)) if worst_z else None,
            "max_adverse_spread_move_inside_trades": float(max(worst_move)) if worst_move else None,
            "days_abs_z_above_default_stop": int((frame.loc[mask, "z"].abs() > inputs.params_default["z_stop"]).sum()),
            "days_deviation_above_default_stop_times_train_sigma": int(
                (deviation[mask].abs() > inputs.params_default["z_stop"] * sigma_train).sum()),
            "chosen_z_stop": stop,
        }
    episode = frame.loc["2024-02-28":"2024-04-16"]
    out["episode_2024_feb_apr"] = {
        "window": "2024-02-28 a 2024-04-16 (valid)",
        "spread_min": float(episode["spread"].min()), "spread_min_date": episode["spread"].idxmin().strftime("%Y-%m-%d"),
        "z_on_2024_02_29": float(frame.at[pd.Timestamp("2024-02-29"), "z"]),
        "z_min_from_2024_03_01": float(episode.loc["2024-03-01":, "z"].min()),
        "sigma_prev_2024_02_28": float(frame.at[pd.Timestamp("2024-02-28"), "sigma_prev"]),
        "sigma_prev_max_in_window": float(episode["sigma_prev"].max()),
    }
    return out
