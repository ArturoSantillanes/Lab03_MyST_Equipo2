# Handoff · Paula · sizing, backtest, métricas y dashboard

Rama: `paula/backtest-dashboard`, creada desde `origin/arturo/kalman2-strategy` (la única rama con el trabajo de Milca y de Arturo). Es para la integración y para que Milca revise el PR.

## Cómo se corre

```
python main.py                    # ... -> strategy -> backtest -> performance (pipeline completo)
streamlit run dashboard/app.py    # solo lee artifacts/; no vuelve a correr nada
python -m pytest                  # o: pytest   (36 pruebas)
```

- Mis dos etapas tardan unos **2.5 minutos**: `backtest` evalúa la rejilla de 2,535 configuraciones en train (≈ 150 s) y `performance` tarda unos segundos.
- El pipeline completo lo domina `cointegration` (Milca). En una máquina de 4 núcleos con OpenBLAS conviene correr con `OPENBLAS_NUM_THREADS=1` (en PowerShell: `$env:OPENBLAS_NUM_THREADS="1"; python main.py`). Solo cambia el número de hilos de BLAS, no el código.

## Qué hice

| Archivo | Qué hace |
|---|---|
| `strategy/sizing.py` | 80% del equity del día repartido según \|β̂_t\|; dólares por pata → acciones enteras con el precio en niveles |
| `backtest/costs.py` | Comisión 0.125% por pata y operación; préstamo 0.25% anual ACT/360 |
| `backtest/engine.py` | Motor día a día multi-pata con efectivo y acciones, libro de órdenes por pata y una fila por operación |
| `backtest/calibration.py` | Grid search solo en train con mínimo de operaciones, regla de meseta y banda de no-rebalanceo |
| `backtest/walk_forward.py` | train → test → valid con parámetros congelados; `ValidationGuard` impide evaluar valid dos veces; diagnóstico del stop |
| `backtest/io.py` | Lectura y escritura de JSON (inf → "inf", NaN → null) |
| `backtest/__init__.py` | Etapa `backtest`: orquesta todo y escribe `artifacts/backtest/` |
| `performance/metrics.py`, `rolling.py`, `figures.py`, `__init__.py` | Métricas por tramo, Sharpe móvil, Deflated Sharpe Ratio y figuras 7 y 8 |
| `dashboard/app.py` | Streamlit + Plotly: vista ejecutiva, vista técnica y calibración |
| `tests/test_accounting.py`, `test_metrics.py`, `test_exposure_cap.py` | Mis tres pruebas |

## Decisiones y por qué

1. **Ejecución:** la señal d_t y los pesos β̂_t se conocen al cierre de t y se ejecutan a ese mismo cierre. d_t gana el cambio de precio de t a t+1, nunca el de t (`backtest/engine.py:85`: el P&L del día usa las acciones de t−1, que se actualizan hasta la línea 124, después de operar).
2. **Capital disponible = equity del día** (marcado al cierre antes de operar), no el capital inicial. Así el tope de 80% se cumple también después de perder.
3. **El tope de 80% se cumple después de pagar la comisión:** N = 0.8·(E − c·(G_actual + 0.8·E)), y las acciones se truncan hacia cero.
4. **Préstamo en ACT/360:** días naturales desde el cierre anterior entre 360 (un fin de semana cobra 3 días), sobre el valor de mercado de las patas cortas del día anterior.
5. **Cada tramo es una corrida independiente:** empieza plana con $1,000,000 y cierra al final del tramo (motivo `fin_de_periodo`). En train empieza el primer día operable, 2015-05-21.
6. **Walk-forward = ancla fija:** calibrar en train, congelar y evaluar train → test → valid. Es un supuesto mío (el enunciado no lo describe) y está en la lista de preguntas al profesor.
7. **Grid search** (`backtest/calibration.py`, procedimiento en el docstring):
   - Objetivo: Sharpe neto de train con rebalanceo diario.
   - Con menos de 30 operaciones el objetivo vale −1e9. **N_MIN = 30 no se bajó**: 675 de las 2,535 configuraciones lo cumplen.
   - Regla de meseta: el mejor es meseta si la mediana de su vecindario (±1 paso por dimensión) queda a lo más 1 error estándar del Sharpe por debajo del suyo (Lo, 2002; ≈ 0.47 con 1,161 días). Si no, se elige el centro de la mejor meseta, que es la configuración viable con la mayor mediana de su vecindario.
8. **Banda de no-rebalanceo:** se elige en train con los umbrales ya congelados (mayor Sharpe de train; en empate, la banda más chica). Política final = la de mayor Sharpe de train entre diaria y banda.

## Supuestos

- Tasa libre de riesgo 0 (el efectivo no paga intereses en el motor) y anualización con 252 días.
- Acciones enteras; sin impacto de mercado ni spread bid-ask más allá de la comisión.
- Sin margen ni colateral explícito por los cortos: el préstamo se cobra con la tasa del enunciado y el efectivo de la venta en corto se queda en caja sin intereses.

## Cifras clave

Parámetros finales (train): **z_entry = 0.5, z_exit = 0.25, z_stop = 3.5, max_hold_days = 150, política de banda 5%**.

| Cifra | train (2015-05-21 a 2019-12-30) | test (2019-12-31 a 2022-12-27) | valid (2022-12-28 a 2025-12-31) |
|---|---|---|---|
| Operaciones | 47 | 37 | 34 |
| Sharpe / Sortino / Calmar | −0.32 / −0.45 / −0.12 | 0.38 / 0.61 / 0.30 | 0.03 / 0.04 / 0.01 |
| Drawdown máximo | −4.80% | −4.63% | −7.60% |
| Win rate / profit factor | 51.1% / 0.49 | 70.3% / 1.72 | 64.7% / 1.02 |
| Rendimiento total | −2.70% | +4.26% | +0.15% |
| P&L bruto | $70,559 | $122,240 | $71,561 |
| Comisiones | −$94,257 | −$77,727 | −$67,875 |
| Préstamo | −$3,299 | −$1,928 | −$2,169 |
| P&L neto | **−$26,997** | **+$42,585** | **+$1,517** |
| Exposición media (todos los días / invertida) | 57.6% / 78.7% | 49.1% / 78.5% | 56.9% / 78.4% |
| Salidas señal / stop / tenencia / fin de tramo | 46 / 0 / 0 / 1 | 37 / 0 / 0 / 0 | 32 / 1 / 0 / 1 |

- Grid search: 2,535 configuraciones, 675 viables. El mejor bruto (0.75 / 0.25 / 3.0 / 150) tiene Sharpe 0.12 con 34 operaciones, pero es **pico aislado** (la mitad de su vecindario no llega a 30 operaciones). La elegida es el centro de la mejor meseta: Sharpe −0.36 con rebalanceo diario y −0.32 con banda, 47 operaciones, y su vecindario sí es meseta. **max_hold_days = 150 está en la orilla**, pero en train no se activa (la operación más larga dura 145 días), así que cualquier valor mayor da el mismo resultado de train.
- Diario contra banda (mismas señales): la banda ahorra $4,811 de comisiones en train, $3,438 en test y $4,086 en valid, y reduce las órdenes por pata de 5,249 a 1,649 en train.
- Deflated Sharpe Ratio de train ≈ 0 con cualquier conteo de pruebas (el Sharpe de train es negativo). El umbral por azar va de 2.08 (910 pruebas de Johansen) a 3.24 (910 × 2,535) en Sharpe anual. PSR fuera de muestra: 0.75 en test y 0.52 en valid.
- Con los valores por defecto de Arturo, train da 6 operaciones, Sharpe 0.33 y +$11,065. No es viable (6 < 30).
- **Cuántas veces cayó un parámetro a un valor por defecto: 0.** El código no tiene ruta de respaldo y ningún parámetro elegido coincide con los de arranque.

## Cuántas veces se evaluó validación

**Una.** Rejilla, regla de meseta, banda y política se fijaron y se commitearon (commit `be41455`) antes de la primera evaluación de test y valid, y no se cambiaron después.

- El pipeline se volvió a correr dos veces: una para guardar los CSV con precisión completa (arreglo de formato) y otra en la verificación final con `python main.py`. Las dos con los mismos parámetros congelados y con cifras idénticas a la primera evaluación.
- `params.json` → `validation_evaluations = 1` en cada corrida (`ValidationGuard` truena si se pide una segunda).
- Nota: al reescribir el autor de mis commits a mi cuenta de GitHub, la fecha de esos commits quedó en la hora de la reescritura. El orden de los commits sí refleja la secuencia real.

## Archivos que produzco (`artifacts/backtest/`)

| Archivo | Contenido |
|---|---|
| `equity.csv` | date, period, equity, cash, long_value, short_value, gross_exposure, exposure_ratio, position, daily_return, drawdown, gross_pnl, commission, borrow_cost |
| `trades.csv` | entry_date, exit_date, period, direction, gross_pnl, commission, borrow_cost, net_pnl, return, holding_days, exit_reason, entry_equity, n_rebalances |
| `costs.csv` | Costos diarios: comisión, préstamo, total, órdenes por pata, valor operado, valor en corto |
| `fills.csv` | Libro de órdenes por pata (entrada, rebalanceo, salida) |
| `positions.csv` | z, posición, días abiertos y motivo de salida con los parámetros finales |
| `metrics.json` | Métricas por tramo, atribución de P&L, exposición, comparación diaria contra banda y DSR |
| `rolling_sharpe.csv` | Sharpe móvil de 252 días y los tramos que abarca cada ventana |
| `params.json` | Parámetros finales, rejilla, regla de meseta, orillas, banda, supuestos y conteo de evaluaciones de valid |
| `grid_search.csv`, `band_grid.csv` | Rejilla completa de train y rejilla de la banda (period = train) |
| `rebalance_comparison.csv` | Diario contra banda por tramo |
| `exit_analysis.json` | Motivos de salida por tramo y verificación del stop sobre \|z\| |
| `artifacts/figures/fig7_curvas_capital.png`, `fig8_distribucion_retornos.png` | Figuras 7 y 8 |

## Límites conocidos

1. **Train pierde neto de costos:** el bruto es positivo (89% de las operaciones ganan en bruto), pero la comisión por operación ($2.0k) supera la ganancia bruta media ($1.5k).
2. El stop sobre |z| casi no actúa: lo confirmo con mis cifras en `docs/secciones/paula.md` §6. La única salida por stop (valid, 2024-02-29) vendió en el peor punto.
3. Muestra chica: 118 operaciones en total, y la peor de cada tramo pesa mucho.
4. max_hold_days queda en la orilla (plano más allá en train). La tenencia máxima no protege con estos parámetros.
5. La selección del grupo se hizo entre 910 pruebas de Johansen: el DSR de train no permite descartar que el resultado sea suerte.
