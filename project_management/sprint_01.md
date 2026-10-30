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
| Paula | Sizing, motor de backtest con costos y calibración | Pendiente |
| Paula | Métricas, Sharpe móvil y dashboard | Pendiente |
| Paula | Pruebas test_accounting, test_metrics, test_exposure_cap | Pendiente |
| Los tres | Integración, reporte (docs/reporte.pdf) y README | Pendiente |

## Notas de bloqueos o desviaciones

- 26 de octubre: ninguno.
- 29 de octubre: el PR de Milca aún no estaba en `main`, así que la rama de Arturo salió de la de Milca.
- 29 de octubre: en Windows, `python main.py` falla en la etapa `data` (`UnicodeDecodeError`: JSON leídos sin `encoding="utf-8"` en archivos de Milca). Milca debe arreglarlo; mientras tanto se corre con `PYTHONUTF8=1`.
- 29 de octubre: la etapa `cointegration` tarda unos 40 minutos en una máquina de 8 núcleos, contra los 6.5 minutos esperados.

El 2 y 3 de noviembre se crea sprint_02.md.
