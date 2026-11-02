"""Lectura y escritura de JSON del bloque de Paula (backtest y performance), en UTF-8 y con JSON estricto.

Dueño: Paula
Lee: Archivos JSON de artifacts/ y data/.
Escribe: Los JSON que le pasen (params.json, exit_analysis.json, metrics.json).
Reglas: inf se escribe como "inf" y NaN como null, porque JSON estricto no admite Infinity ni NaN.
"""
import json
from pathlib import Path

import numpy as np


def read_json(path: Path) -> dict:
    """Lee un JSON en UTF-8."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def json_safe(value):
    """Convierte inf/NaN y tipos de numpy a algo que JSON estricto acepte (inf → 'inf', NaN → None)."""
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else ("inf" if np.isinf(value) else float(value))
    return value


def write_json(path: Path, content: dict) -> None:
    """Escribe un JSON en UTF-8 con sangría."""
    Path(path).write_text(json.dumps(json_safe(content), indent=2, ensure_ascii=False), encoding="utf-8")
