"""Calibra umbrales (z_entry, z_exit, z_stop, tenencia máxima) con grid search.

Dueño: Paula
Lee: Señales de Arturo y el motor de backtest.
Escribe: artifacts/backtest/grid_search.csv, params.json
Reglas: SOLO train. Si el óptimo cae en el borde de la rejilla, decirlo.
"""
