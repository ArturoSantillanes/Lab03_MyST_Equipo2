"""Costos: comisión por pata y préstamo sobre el nocional en corto (declarar base de días).

Dueño: Paula
Lee: Posiciones por pata.
Escribe: Tabla de costos.
Reglas: Comisión 0.125% por operación y por pata; préstamo 0.25% anual, devengo diario.

Convenciones declaradas:
- Comisión: 0.125% del valor operado |Δacciones|·precio de CADA pata en CADA operación (entrada, rebalanceo y salida).
- Préstamo: 0.25% anual sobre el valor de mercado de las patas cortas al cierre anterior, con devengo diario en base
  ACT/360 (convención de mercado de dinero en EE. UU.): cada día hábil se cobran los días NATURALES transcurridos desde
  el cierre anterior entre 360. Un fin de semana cobra 3 días. La misma base se usa en todo el proyecto.
"""
import numpy as np

COMMISSION_RATE = 0.00125
BORROW_RATE_ANNUAL = 0.0025
DAY_COUNT_BASIS = 360


def commission(traded_value: np.ndarray | float) -> np.ndarray | float:
    """Comisión de una operación por pata: 0.125% del valor absoluto operado."""
    return COMMISSION_RATE * np.abs(traded_value)


def borrow_accrual(short_value: float, calendar_days: int) -> float:
    """Devengo del préstamo de un día: tasa anual × valor en corto (positivo) × días naturales / 360."""
    return BORROW_RATE_ANNUAL * short_value * calendar_days / DAY_COUNT_BASIS
