"""Filtro de Kalman 2: estima media y varianza latentes del spread y calcula el z-score.

Dueño: Arturo
Lee: artifacts/kalman/spread_t.csv
Escribe: artifacts/kalman/gain_2.csv, posvar_2.csv, z_t.csv
Reglas: El z-score usa la media y la varianza PREVIAS a la actualización, nunca la posterior. Q y R solo con train.
"""


def run() -> None:
    """Ejecuta esta etapa del pipeline (pendiente de implementar)."""
    raise NotImplementedError("Lo implementa Arturo")
