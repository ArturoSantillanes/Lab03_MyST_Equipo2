"""Biblioteca de tamaño de posición: 80% del capital disponible repartido según el vector de pesos. La usa el backtest.

Dueño: Paula
Lee: Vector β̂_t y capital disponible.
Escribe: Nada (solo funciones).
Reglas: 80% del capital, pesos según β; el signo lo da la dirección de la operación.

Decisiones declaradas:
- Capital disponible = EQUITY DEL DÍA (valor del portafolio marcado al cierre de t, antes de operar), no el capital
  inicial: así la exposición escala con ganancias y pérdidas y el tope de 80% se cumple también después de perder.
- El nocional objetivo descuenta de antemano una cota de la comisión de la operación, para que DESPUÉS de pagarla la
  exposición bruta siga ≤ 80% del equity: N = 0.8·(E − c·(G_actual + 0.8·E)). La cota usa que el valor operado no
  puede exceder G_actual + N (lo que se cierra más lo que se abre). Cuesta ≈0.1% del nocional.
- Los pesos son EXPOSICIÓN EN VALOR (el proyecto usa log-precios): primero dólares por pata, w_i = β̂_i/Σ|β̂_j|, y luego
  acciones = dólares / precio en niveles. Las acciones se truncan hacia cero (acciones enteras), lo que solo puede
  reducir la exposición.
- Signo: largo el spread (d = +1) = comprar cada activo con β̂_i > 0 y vender cada uno con β̂_i < 0; corto el
  spread (d = −1) invierte todos los signos.
"""
import numpy as np

INITIAL_CAPITAL = 1_000_000.0
EXPOSURE_CAP = 0.80


def leg_weights(beta: np.ndarray) -> np.ndarray:
    """Pesos con signo por pata: β̂_i / Σ_j |β̂_j| (suman 1 en valor absoluto)."""
    beta = np.asarray(beta, dtype=float)
    return beta / np.abs(beta).sum()


def target_gross(equity: float, current_gross: float, commission_rate: float, cap: float = EXPOSURE_CAP) -> float:
    """Nocional bruto objetivo: 80% del equity del día descontando la cota de comisión de la operación."""
    return max(cap * (equity - commission_rate * (current_gross + cap * equity)), 0.0)


def target_values(direction: int, beta: np.ndarray, gross: float) -> np.ndarray:
    """Dólares con signo por pata: d·w_i·N (positivo = largo en el activo, negativo = corto)."""
    return direction * leg_weights(beta) * gross


def target_shares(values: np.ndarray, prices: np.ndarray) -> np.ndarray:
    """Acciones enteras por pata, truncadas hacia cero (nunca exceden el valor objetivo)."""
    return np.trunc(np.asarray(values, dtype=float) / np.asarray(prices, dtype=float))


def gross_exposure(shares: np.ndarray, prices: np.ndarray) -> float:
    """Exposición bruta: Σ |acciones_i·precio_i|."""
    return float(np.abs(np.asarray(shares) * np.asarray(prices)).sum())


def needs_rebalance(current_values: np.ndarray, desired_values: np.ndarray, band: float) -> bool:
    """Política de banda: rebalancear si alguna pata se aleja de su objetivo más que `band` del nocional objetivo.

    La desviación de la pata i es |V_i − V*_i| / Σ_j |V*_j|. band = 0 equivale a rebalanceo diario.
    """
    total = np.abs(desired_values).sum()
    if total == 0:
        return bool(np.abs(current_values).sum() > 0)
    return bool(np.max(np.abs(np.asarray(current_values) - desired_values)) / total > band)
