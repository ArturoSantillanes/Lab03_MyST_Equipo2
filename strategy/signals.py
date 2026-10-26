"""Máquina de estados de señales: entrada, salida, stop sobre abs(z), tenencia máxima y motivo de salida.

Dueño: Arturo
Lee: Serie z y parámetros (z_entry, z_exit, z_stop, max_hold_days).
Escribe: Señal diaria en {-1, 0, +1} con motivo de salida.
Reglas: Parámetros calibrados SOLO con train. Tenencia máxima calibrada a la vida media.
"""
