"""Máquina de estados de señales sobre el z-score: entrada, salida, stop sobre |z| y tenencia máxima.

Dueño: Arturo
Lee: serie z (artifacts/kalman/z_t.csv, columna z) y parámetros (z_entry, z_exit, z_stop, max_hold_days)
Escribe: (lo guarda strategy.run) artifacts/strategy/params_default.json, positions_default.csv, direction_check.json
Reglas: los parámetros definitivos se calibran SOLO con train (grid search de Paula con su motor, usando
generate_positions). Los de params_default.json son valores de arranque. Sin look-ahead.

Regla de la diapositiva del profesor (h_t = días que lleva abierta la posición):
    d_t = +1 (largo el spread)  si d_{t−1} = 0 y z_t < −z_in
    d_t = −1 (corto el spread)  si d_{t−1} = 0 y z_t > +z_in
    d_t =  0 (cerrar)           si |z_t| < z_out, o |z_t| > z_stop, o h_t ≥ h_max
    d_t = d_{t−1} (mantener)    en cualquier otro caso
Las entradas solo ocurren desde plano: para voltear primero se cierra. Si en plano z_t pasa de z_stop, las
condiciones de entrar y de cerrar se cumplen a la vez; manda la de cerrar (no se abre una posición que el stop
cerraría ese mismo día). Motivo de salida cuando coinciden: stop, luego señal, luego tenencia máxima.
d_t se decide con el cierre del día t y se aplica del cierre de t al de t+1 (lo hace el backtest).
z faltante (burn-in) = no hay señal: no se abre posición.

Dirección: largo el spread = comprar cada activo con β̂_{t,i} > 0 y vender cada uno con β̂_{t,i} < 0, en proporción
a |β̂_{t,i}| (exposición en valor, log-precios). Corto el spread = todos los signos al revés.
"""
import math

import numpy as np
import pandas as pd

DEFAULT_Z_ENTRY = 2.0
DEFAULT_Z_EXIT = 0.5
DEFAULT_Z_STOP = 3.0


def default_params(half_life_days: float) -> dict:
    """Valores de arranque; max_hold_days = días que tarda el desequilibrio en bajar de z_entry a z_exit.

    Con reversión exponencial, el desequilibrio se reduce a la mitad cada vida media, así que pasar de z_entry a z_exit
    toma vida_media·log2(z_entry/z_exit) días. Si para entonces no revirtió, la posición "dejó de revertir".
    """
    hold = half_life_days * math.log2(DEFAULT_Z_ENTRY / DEFAULT_Z_EXIT)
    return {"z_entry": DEFAULT_Z_ENTRY, "z_exit": DEFAULT_Z_EXIT, "z_stop": DEFAULT_Z_STOP,
            "max_hold_days": int(math.ceil(hold)), "half_life_days": half_life_days,
            "max_hold_rule": "ceil(vida_media · log2(z_entry / z_exit))"}


def exit_reason(z: float, held: int, params: dict) -> str:
    """Motivo de cierre del día ('' si se mantiene la posición)."""
    if abs(z) > params["z_stop"]:
        return "stop"
    if abs(z) < params["z_exit"]:
        return "senal"
    if held >= params["max_hold_days"]:
        return "tenencia_maxima"
    return ""


def entry_direction(z: float, params: dict) -> int:
    """+1 / −1 si desde plano se abre largo / corto el spread; 0 si no (incluye z faltante o más allá del stop)."""
    if np.isnan(z) or abs(z) > params["z_stop"]:
        return 0
    if z < -params["z_entry"]:
        return 1
    if z > params["z_entry"]:
        return -1
    return 0


def generate_positions(z: pd.Series, params: dict) -> pd.DataFrame:
    """Señal diaria del spread en {−1, 0, +1}.

    Parámetros: z (serie de z-score por fecha) y params con z_entry, z_exit, z_stop y max_hold_days.
    Devuelve, por fecha: position (d_t), holding_days (h_t: días abierta al cierre de t; 0 en plano) y exit_reason
    ('senal', 'stop' o 'tenencia_maxima' el día que se cierra; '' en cualquier otro día).
    """
    position, held = 0, 0
    rows = []
    for value in z.to_numpy(dtype=float):
        reason = ""
        if position != 0:
            held += 1
            reason = exit_reason(value, held, params)
            if reason:
                position, held = 0, 0
        else:
            position = entry_direction(value, params)
        rows.append((position, held, reason))
    return pd.DataFrame(rows, index=z.index, columns=["position", "holding_days", "exit_reason"])


def trades_table(positions: pd.DataFrame) -> pd.DataFrame:
    """Una fila por operación: entrada, salida, dirección, días hábiles abierta y motivo de salida ('abierta' si no
    cerró dentro de las fechas recibidas)."""
    previous = positions["position"].shift(fill_value=0)
    entries = positions.index[(previous == 0) & (positions["position"] != 0)]
    exits = positions.index[positions["exit_reason"].fillna("") != ""]  # al leer el CSV, '' llega como NaN
    rows = []
    for entry in entries:
        later = exits[exits > entry]
        exit_date = later[0] if len(later) else None
        last = exit_date if exit_date is not None else positions.index[-1]
        rows.append({"entry": entry, "exit": exit_date, "direction": int(positions.at[entry, "position"]),
                     "days": positions.index.get_loc(last) - positions.index.get_loc(entry),
                     "exit_reason": positions.at[exit_date, "exit_reason"] if exit_date is not None else "abierta"})
    return pd.DataFrame(rows)


def spread_leg_returns(log_prices: pd.DataFrame, beta_t: pd.DataFrame) -> pd.Series:
    """Rendimiento del día siguiente de una posición LARGA en el spread por unidad de nocional bruto.

    r_{t+1} = Σ_i β̂_{t,i}·(log P_{t+1,i} − log P_{t,i}) / Σ_i |β̂_{t,i}|  (pesos fijos del día t).
    """
    next_change = log_prices[beta_t.columns].diff().shift(-1)
    return (beta_t * next_change).sum(axis=1, min_count=1) / beta_t.abs().sum(axis=1)


def direction_check(positions: pd.DataFrame, leg_returns: pd.Series, z: pd.Series, period: pd.Series) -> dict:
    """Verificación del signo en train: rendimiento bruto por operación con la regla y con todos los signos al revés.

    No es un resultado de desempeño (no hay costos ni sizing; eso es del backtest): solo confirma que, con la
    convención de dirección, las posiciones ganan cuando el spread revierte.
    """
    train = period == "train"
    gross = positions["position"] * leg_returns
    trades = trades_table(positions[train])
    trades["gross_log_return"] = [float(gross.loc[t.entry:t.exit].sum()) if t.exit is not None else np.nan
                                  for t in trades.itertuples()]
    reverting = pd.concat([z, leg_returns], axis=1).loc[train].dropna()
    return {
        "period": "train",
        "corr_z_vs_next_day_spread_return": float(np.corrcoef(reverting.iloc[:, 0], reverting.iloc[:, 1])[0, 1]),
        "train_trades": len(trades),
        "train_trades_with_positive_gross": int((trades["gross_log_return"] > 0).sum()),
        "train_gross_log_return_rule": float(trades["gross_log_return"].sum()),
        "train_gross_log_return_reversed_signs": float(-trades["gross_log_return"].sum()),
        "trades": trades.assign(entry=pd.to_datetime(trades["entry"]).dt.strftime("%Y-%m-%d"),
                                exit=pd.to_datetime(trades["exit"]).dt.strftime("%Y-%m-%d")).to_dict(orient="records"),
    }
