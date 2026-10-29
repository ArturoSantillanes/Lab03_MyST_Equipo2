"""Etapa 1 del embudo: universo de candidatos y su justificación económica.

Dueño: Milca
Lee: data/universe.json (configuración: sector, ventana y los 9 activos con su justificación).
Escribe: Nada; expone el universo a las demás etapas.
Reglas: El universo NO está escrito en el código; se lee del archivo de configuración.
La justificación económica (mismo sector, tendencia estocástica común) es la defensa
frente a falsos positivos: más activos probados = más falsos positivos.
"""
import json
from pathlib import Path

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"


def load_universe(path: Path = UNIVERSE_PATH) -> dict:
    """Lee el archivo de configuración del universo."""
    return json.loads(path.read_text())


def candidate_tickers(universe: dict) -> list[str]:
    """Tickers del universo, en minúsculas y en el orden del archivo."""
    return [asset["ticker"] for asset in universe["assets"]]
