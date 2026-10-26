"""Primer eigenvector de Johansen móvil en el tiempo.

Dueño: Milca
Lee: data/processed/prices.csv, data/processed/splits.json
Escribe: artifacts/cointegration/eigvec_rolling.csv
Reglas: Cada fecha usa solo datos hasta esa fecha (sin look-ahead). Alimenta la figura 4.
"""
