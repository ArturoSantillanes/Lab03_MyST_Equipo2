"""Etapa 'data' del pipeline: descarga, limpia y parte los datos. Orquesta download, clean y split.

Dueño: Milca
Lee: Fuente externa de precios diarios (la libreria la elige Milca).
Escribe: data/processed/prices.csv, data/processed/splits.json
Reglas: Datos diarios de 10 a 15 años; partición 60/20/20 cronológica sin look-ahead; semilla 42 (ver main.py).
"""


def run() -> None:
    """Ejecuta esta etapa del pipeline (pendiente de implementar)."""
    raise NotImplementedError("Lo implementa Milca")
