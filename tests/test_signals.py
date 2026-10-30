"""Prueba de Arturo: la máquina de estados respeta la regla del profesor sobre series sintéticas de z.

Cada serie está construida a propósito para disparar un caso: entrada desde plano, salida por z_exit, stop por
z_stop, tenencia máxima por max_hold_days, no voltear directamente, no entrar más allá del stop y z faltante.
"""
import numpy as np
import pandas as pd

from strategy.signals import default_params, generate_positions

PARAMS = {"z_entry": 2.0, "z_exit": 0.5, "z_stop": 3.0, "max_hold_days": 4}


def positions(values: list[float]) -> pd.DataFrame:
    """generate_positions sobre una serie sintética con fechas hábiles."""
    return generate_positions(pd.Series(values, index=pd.bdate_range("2020-01-01", periods=len(values))), PARAMS)


def test_long_entry_from_flat_and_exit_by_signal():
    result = positions([0.0, -2.5, -1.5, -0.3, -2.1])
    assert result["position"].tolist() == [0, 1, 1, 0, 1]
    assert result["exit_reason"].tolist() == ["", "", "", "senal", ""]


def test_short_entry_and_exit_by_signal():
    result = positions([2.4, 1.0, 0.2, 0.0])
    assert result["position"].tolist() == [-1, -1, 0, 0]
    assert result["exit_reason"].iloc[2] == "senal"


def test_no_entry_inside_the_band():
    result = positions([1.9, -1.9, 2.0, -2.0, 0.0])
    assert (result["position"] == 0).all()


def test_stop_closes_when_abs_z_exceeds_z_stop():
    result = positions([-2.2, -2.8, -3.2, -2.5])
    assert result["position"].tolist() == [1, 1, 0, 1]
    assert result["exit_reason"].iloc[2] == "stop"


def test_max_holding_period_closes_the_position():
    result = positions([2.5, 2.4, 2.3, 2.2, 2.1, 1.0])
    assert result["position"].tolist() == [-1, -1, -1, -1, 0, 0]
    assert result["holding_days"].tolist() == [0, 1, 2, 3, 0, 0]
    assert result["exit_reason"].iloc[4] == "tenencia_maxima"


def test_cannot_flip_directly_from_short_to_long():
    # z salta de +2.5 a −2.5 sin caer en |z| < z_exit: ninguna condición de cierre se cumple, así que se mantiene
    # el corto (la regla solo abre desde plano) hasta que la tenencia máxima lo cierra.
    result = positions([2.5, -2.5, -2.6, -2.4, -2.2, -2.3])
    assert result["position"].tolist() == [-1, -1, -1, -1, 0, 1]
    assert result["exit_reason"].iloc[4] == "tenencia_maxima"


def test_short_to_long_crossing_closes_first_by_signal():
    result = positions([2.5, 0.1, -2.5])
    assert result["position"].tolist() == [-1, 0, 1]
    assert result["exit_reason"].iloc[1] == "senal"


def test_no_entry_beyond_the_stop_level():
    result = positions([-3.5, 3.4, -2.5])
    assert result["position"].tolist() == [0, 0, 1]


def test_missing_z_never_opens_a_position():
    result = positions([np.nan, np.nan, -2.5])
    assert result["position"].tolist() == [0, 0, 1]


def test_default_max_hold_is_calibrated_to_the_half_life():
    params = default_params(15.3)
    assert params["max_hold_days"] == int(np.ceil(15.3 * np.log2(params["z_entry"] / params["z_exit"])))
