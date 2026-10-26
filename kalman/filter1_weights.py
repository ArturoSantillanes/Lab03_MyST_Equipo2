"""Filtro de Kalman 1: estima los pesos dinámicos β_t y el spread S_t = β̂_tᵀ P_t.

Dueño: Milca
Lee: data/processed/prices.csv, artifacts/cointegration/selected_group.json
Escribe: artifacts/kalman/beta_t.csv, spread_t.csv, gain_1.csv, posvar_1.csv
Reglas: Q y R se eligen SOLO con train. Normalizar sobre una pata y declarar signo. Declarar inicialización y burn-in.
"""


def run() -> None:
    """Ejecuta esta etapa del pipeline (pendiente de implementar)."""
    raise NotImplementedError("Lo implementa Milca")
