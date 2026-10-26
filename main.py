"""Pipeline completo del Lab 03 (arbitraje estadístico con cointegración y filtros de Kalman).

Un solo comando: python main.py

Corre las etapas en este orden y cada una expone una función run():
    data -> cointegration -> kalman.filter1_weights -> kalman.filter2_spread
         -> strategy -> backtest -> performance

Para cada etapa imprime su nombre y su duración. Se detiene en el primer error
e indica qué etapa falló. Nadie necesita editar este archivo: cada dueño
implementa el run() de sus módulos.

Semilla: SEED = 42, fijada para random y numpy (reproducibilidad).
"""
import importlib
import random
import sys
import time

import numpy as np

SEED = 42

STAGES = [
    "data",
    "cointegration",
    "kalman.filter1_weights",
    "kalman.filter2_spread",
    "strategy",
    "backtest",
    "performance",
]


def set_seed(seed: int = SEED) -> None:
    """Fija la semilla de random y numpy."""
    random.seed(seed)
    np.random.seed(seed)


def run_stage(name: str) -> float:
    """Importa la etapa, ejecuta su run() y devuelve la duración en segundos."""
    start = time.perf_counter()
    importlib.import_module(name).run()
    return time.perf_counter() - start


def main() -> int:
    """Ejecuta todas las etapas en orden; devuelve 0 si todo salió bien, 1 si una falló."""
    set_seed()
    print(f"Semilla fijada: {SEED}")
    for name in STAGES:
        print(f"[{name}] iniciando...")
        try:
            seconds = run_stage(name)
        except Exception as error:
            print(f"ERROR en la etapa '{name}': {type(error).__name__}: {error}", file=sys.stderr)
            return 1
        print(f"[{name}] terminada en {seconds:.2f} s")
    print("Pipeline completo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
