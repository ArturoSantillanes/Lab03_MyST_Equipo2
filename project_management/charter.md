# Project charter · Lab 03 · Equipo 2

## 1. Alcance

**Sí hará:**
- Estrategia long/short sobre un grupo de 6 activos cointegrados, elegidos solo con la partición de entrenamiento.
- Datos diarios de 10 a 15 años, partición cronológica 60/20/20 (train/test/valid).
- Johansen, filtros de Kalman 1 y 2, VECM y señales por z-score con stop y tenencia máxima.
- Posición de 80% del capital repartida según el vector de pesos.
- Backtest con comisión 0.125% por pata (entrada y salida) y préstamo 0.25% anual; capital inicial $1,000,000.
- Dashboard, pruebas (pytest) y reporte.

**No hará:** Nivel B (varios grupos con Risk Parity), operación intradía, ejecución en vivo, ni elegir parámetros mirando prueba o validación.

## 2. Plan de hitos

| Hito | Entregable | Fecha | Responsable |
|---|---|---|---|
| M0 | Repositorio y estructura listos | 26 oct | Arturo |
| M1 | Pipeline de datos completo | 26 oct | Milca |
| M2 | Cointegración completa y grupo elegido | 27 oct | Milca |
| M3 | Filtro 1 validado | 27 oct | Milca |
| M4 | Filtro 2, VECM y señales | 29 oct | Arturo |
| M5 | Backtest completo | 31 oct | Paula |
| M6 | Dashboard funcionando | 31 oct | Paula |
| M7 | Integración, reporte y README | 1 nov | Los tres |
| M8 | Último commit antes de las 23:59 | 2 nov | Los tres |
| M9 | Colchón y ensayos de la exposición | 3 nov | Los tres |

Nota: el repositorio se evalúa con el último commit anterior a las 23:59 del día previo a la exposición.

## 3. Registro de riesgos

| # | Riesgo | Probabilidad | Impacto | Mitigación |
|---|---|---|---|---|
| 1 | Ningún grupo pasa Johansen y ADF | Media | Alto | Universo amplio con justificación económica; ampliar el universo antes de relajar criterios; reportar honestamente si no hay grupo |
| 2 | Pérdida de cointegración fuera de muestra | Alta | Alto | Stop sobre abs(z) y tenencia máxima; medir ADF del spread en test y valid; reportarlo como límite |
| 3 | El filtro de Kalman diverge o queda mal calibrado | Media | Alto | Q y R justificadas solo con train; prueba de innovaciones normalizadas; revisar varianza posterior |
| 4 | Falta de tiempo para dashboard y reporte | Media | Alto | Hitos con fechas; formato de archivos fijado en contrato.md; colchón el 3 nov |
| 5 | Commits o dominio del tema desbalanceados | Media | Alto | Una rama por persona, PR revisado por otra persona; preguntas tipo profe al terminar cada módulo |
| 6 | Enfermedad o ausencia de un integrante | Baja | Medio | Handoff escrito por bloque; todos pueden explicar cualquier parte; colchón de fechas |

## 4. Roles y referencia a Git

| Integrante | Responsabilidad | Pruebas | Figuras | Rama |
|---|---|---|---|---|
| Milca | `data/`, `cointegration/`, `kalman/filter1_weights.py` | test_split, test_cointegration, test_beta_stability | 1, 3 y 4 | `milca/*` |
| Arturo | Repositorio y `project_management/`, `kalman/filter2_spread.py`, `strategy/vecm.py`, `zscore.py`, `signals.py` | test_causality, test_innovations, test_signals | 2, 5 y 6 | `arturo/*` |
| Paula | `strategy/sizing.py`, `performance/`, `backtest/`, `dashboard/` | test_accounting, test_metrics, test_exposure_cap | 7 y 8 | `paula/*` |

Revisión cruzada de pull requests: Arturo revisa a Milca, Paula revisa a Arturo, Milca revisa a Paula. Cada quien edita solo sus archivos. `main` queda protegida por acuerdo: se hace merge solo cuando el bloque está terminado y verificado.
