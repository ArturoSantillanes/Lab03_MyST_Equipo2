"""Etapa 1 del embudo: sectores candidatos, sus activos y su justificación económica.

Dueño: Milca
Lee: data/universe.json (preregistro v5: sectores, activos con justificación, protocolo y regla de selección).
Escribe: Nada; expone el universo a las demás etapas.
Reglas: El universo NO está escrito en el código; se lee del archivo de configuración, que se registró en git
antes de descargar datos. La justificación económica (mismo sector, tendencia estocástica común) es la defensa
frente a falsos positivos: más activos probados = más falsos positivos.
"""
import json
from pathlib import Path

UNIVERSE_PATH = Path(__file__).resolve().parent.parent / "data" / "universe.json"


def load_universe(path: Path = UNIVERSE_PATH) -> dict:
    """Lee el archivo de configuración del universo."""
    return json.loads(path.read_text(encoding="utf-8"))


def sector_tickers(sector: dict) -> list[str]:
    """Tickers de un sector, en minúsculas y en el orden del archivo."""
    return [asset["ticker"] for asset in sector["assets"]]
