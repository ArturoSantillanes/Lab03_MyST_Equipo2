# Sprint 01 · semana del 26 oct al 1 nov

## Objetivo

Dejar el repositorio y los documentos de gestión listos, y construir en orden los bloques del pipeline: datos y cointegración (Milca), filtros, VECM y señales (Arturo), y backtest, métricas y dashboard (Paula). Al cierre de la semana el pipeline corre completo con `python main.py` y el dashboard lee `artifacts/`.

## Tareas

| Persona | Tarea | Estado |
|---|---|---|
| Arturo | Armado del repositorio, estructura, esqueletos y documentos de gestión | Hecho |
| Milca | Pipeline de datos (descarga, limpieza, partición 60/20/20) | Hecho (rama `milca/cointegracion-kalman1`) |
| Milca | Selección de grupo: correlación, Engle-Granger, Johansen, validación del spread | Hecho |
| Milca | Filtro de Kalman 1 y pruebas test_split, test_cointegration, test_beta_stability | Hecho |
| Arturo | `pytest.ini`, yfinance en requirements, alcance en README y charter | Hecho |
| Arturo | Filtro de Kalman 2, VECM, z-score y señales | Hecho (rama `arturo/kalman2-strategy`) |
| Arturo | Pruebas test_causality, test_innovations, test_signals (y test_filter2_step) | Hecho |
| Arturo | Figuras 2, 5 y 6; handoff y sección del reporte | Hecho |
| Paula | Sizing, motor de backtest con costos y calibración | Hecho (rama `paula/backtest-dashboard`) |
| Paula | Métricas, Sharpe móvil y dashboard | Hecho (rama `paula/backtest-dashboard`) |
| Paula | Pruebas test_accounting, test_metrics, test_exposure_cap | Hecho (rama `paula/backtest-dashboard`) |
| Los tres | Integración de las ramas en `main` y README | Hecho (PR #1 y #2, 1 de noviembre) |
| Los tres | Reporte (docs/reporte.pdf) | Pasa a sprint_02 |

## Notas de bloqueos o desviaciones

- 26 de octubre: ninguno.
- 29 de octubre: el PR de Milca aún no estaba en `main`, así que la rama de Arturo salió de la de Milca.
- 29 de octubre: en Windows, `python main.py` falla en la etapa `data` (`UnicodeDecodeError`: JSON leídos sin `encoding="utf-8"` en archivos de Milca). Milca debe arreglarlo; mientras tanto se corre con `PYTHONUTF8=1`. **Resuelto el 30 de octubre** (commit `2746c51`): ya corre con cp1252 sin variables especiales.
- 29 de octubre: la etapa `cointegration` tarda unos 40 minutos en una máquina de 8 núcleos con Windows, contra los 6.5 minutos reportados en Mac. El 3 de noviembre midió de 37 a 39 minutos, con 42 y 44 minutos para el pipeline completo.

- 2 de noviembre: Paula terminó sizing, backtest, calibración, métricas, dashboard y sus 3 pruebas; el pipeline corre completo hasta `performance` y hay 36 pruebas. La estrategia pierde neto de costos en train y queda en cero en valid (ver handoff_paula.md). Desde esta sesión no hubo permiso de escritura en GitHub: la rama se sube desde la máquina de Paula.

Lo que sigue está en [sprint_02.md](sprint_02.md).
