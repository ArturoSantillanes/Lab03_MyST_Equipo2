# Tamaño de posición, backtest, métricas y dashboard (Paula)

Código:
- `strategy/sizing.py`;
- `backtest/` (`engine.py`, `costs.py`, `calibration.py`, `walk_forward.py`, `io.py`);
- `performance/` (`metrics.py`, `rolling.py`, `figures.py`);
- `dashboard/app.py`.

Pruebas: `tests/test_accounting.py`, `test_metrics.py` y `test_exposure_cap.py`.

Toda cifra indica su tramo:
- **train:** operable del 2015-05-21 (fin del burn-in del filtro 2) al 2019-12-30, 1,161 días;
- **test:** del 2019-12-31 al 2022-12-27, 754 días;
- **valid:** del 2022-12-28 al 2025-12-31, 755 días.

Lo que recibo:
- los pesos β̂_t del filtro 1 (`beta_t.csv`, XEL = +1);
- el z-score del filtro 2 con estadísticos previos (`z_t.csv`, solo donde `burn_in == False`);
- los precios ajustados en niveles de los 6 activos del grupo (`prices.csv`);
- la máquina de estados de Arturo, `strategy.generate_positions`.

Todo lo que estimo o elijo usa solo train.

## 1. Métricas (`performance/metrics.py`)

| Métrica | Definición |
|---|---|
| Rendimiento diario | r_t = E_t/E_{t−1} − 1 (el primero contra el capital inicial) |
| Sharpe | media(r)/desv(r)·√252 (desviación muestral, ddof = 1) |
| Sortino | media(r)/√(media(min(r, 0)²))·√252 (desviación a la baja sobre todos los días) |
| Drawdown máximo | mín_t E_t / máx_{s≤t}(E_s, capital inicial) − 1 |
| Calmar | CAGR/\|drawdown máximo\|, con CAGR = (E_final/E_0)^(252/n) − 1 |
| Win rate | fracción de operaciones con P&L neto > 0 |
| Profit factor | Σ ganancias netas / \|Σ pérdidas netas\| |
| Sharpe móvil | ventana de 252 días sobre los tres tramos encadenados (`rolling_sharpe.csv` indica los tramos de cada ventana) |

- **Tasa libre de riesgo = 0.** El motor no paga intereses sobre el efectivo y la estrategia es long/short casi autofinanciada, así que su rendimiento ya es un rendimiento en exceso.
- **Factor de anualización = 252.**
- Cada tramo se calcula por separado. `tests/test_metrics.py` verifica cada fórmula contra curvas sintéticas de resultado conocido en forma cerrada (por ejemplo, rendimientos que alternan +1% y −0.5%). El Sharpe máximo esperado por azar se verifica contra una simulación Monte Carlo.

## 2. Tamaño de posición (`strategy/sizing.py`)

1. **Capital disponible = equity del día** (marcado al cierre de t antes de operar). Lo elegí sobre el capital inicial para que la exposición escale con las ganancias y las pérdidas y el tope se cumpla también después de perder.
2. **Nocional bruto objetivo:** N = 0.8·(E − c·(G_actual + 0.8·E)), con c = 0.125%.
   - El término de la comisión es una cota del costo de la propia operación, porque lo operado no excede lo que se cierra más lo que se abre.
   - Así, después de pagarla, la exposición sigue ≤ 80% del equity. Cuesta ≈ 0.1% del nocional.
3. **Reparto:** la pata i recibe d_t·(β̂_{t,i}/Σ_j|β̂_{t,j}|)·N dólares.
   - El signo sigue la convención de Milca y Arturo: largo el spread (d = +1) compra cada activo con β̂_i > 0 y vende cada uno con β̂_i < 0; corto el spread invierte todos los signos.
   - Como el spread es de log-precios, los pesos son **exposición en valor**. Primero se calculan los dólares por pata y luego acciones = dólares/precio en niveles, truncadas hacia cero (acciones enteras), lo que solo reduce la exposición.
4. **Rebalanceo:**
   - **Diario:** cada día a los pesos β̂_t del día.
   - **Banda:** solo si alguna pata se aleja de su objetivo más que la banda (|V_i − V*_i| / Σ|V*|), o si la exposición bruta pasa de 80%. Este tope se aplica siempre y con cualquier banda.

`tests/test_exposure_cap.py` verifica que la exposición bruta nunca pasa de 80% del equity:
- en los 2,670 días de los tres tramos, recalculando desde el libro de órdenes;
- en un escenario sintético adverso, donde las patas cortas suben 90% y el tope obliga a reducir;
- en 500 casos aleatorios de la función de sizing.

## 3. Motor de backtest y costos (`backtest/engine.py`, `costs.py`)

**Supuesto de ejecución:** la señal d_t y los pesos β̂_t se conocen al cierre de t y se ejecutan a ese precio de cierre. d_t gana el cambio de precio de t a t+1 y nunca el de t. Cada día:

1. P&L bruto = Σ_i acciones_{t−1,i}·(P_{t,i} − P_{t−1,i}), con las acciones de ANTES de operar (`engine.py:85`).
2. Préstamo = 0.25% × valor en corto al cierre de t−1 × días naturales / 360.
3. Se opera la diferencia hacia las acciones objetivo y se paga 0.125% del valor operado de cada pata.

**Detalles de la contabilidad:**
- **Base de días del préstamo: ACT/360**, la convención de mercado de dinero en EE. UU. Un viernes → lunes cobra 3 días, y la misma base se usa en todo el proyecto.
- **Contabilidad en efectivo:** comprar resta efectivo y vender en corto lo suma; valor del portafolio = efectivo + Σ acciones_i·P_i.
- **Comisión** en la entrada, en cada rebalanceo y en la salida, por pata.
- **Cada tramo es una corrida independiente** que empieza plana con $1,000,000 y cierra lo abierto el último día (motivo `fin_de_periodo`).
- **Cada operación registra:** entrada, salida, dirección, P&L bruto, comisiones, préstamo, P&L neto, retorno (sobre el equity de entrada), días y motivo.

`tests/test_accounting.py` (la prueba obligatoria) verifica dos cosas **día a día**, sobre datos sintéticos (con un β que cambia de signo y ambas políticas), sobre train real y sobre los artefactos de los tres tramos:
1. El valor del portafolio es igual al efectivo más el valor de todas las patas, reconstruidos de forma independiente desde el libro de órdenes, con tolerancia de $10⁻⁶.
2. El costo total registrado es igual a Σ de 0.125% × valor operado de cada orden por pata, más el devengo del préstamo recalculado desde las acciones reconstruidas.

## 4. Calibración (`backtest/calibration.py`)

**Procedimiento:**
- Rejilla completa de **2,535 configuraciones** con 0 < z_exit < z_entry < z_stop:
  - z_entry ∈ {0.25, …, 2.5} (10 valores);
  - z_exit ∈ {0.1, 0.25, 0.5, 0.75, 1.0};
  - z_stop ∈ {2.5, 3, 3.5, 4, ∞};
  - max_hold_days ∈ {5, 10, 15, 20, 25, 31, 40, 50, 60, 80, 100, 125, 150}.
- Cada configuración pasa por `generate_positions` y el motor con costos, **solo en train**.
- Objetivo: Sharpe neto de train.
- La rejilla completa está en `artifacts/backtest/grid_search.csv`.

**Restricción de operaciones mínimas:** con menos de N_MIN = 30 operaciones el objetivo vale −1e9.
- Es indispensable. Sin ella, el mejor de la rejilla tiene 7 operaciones y Sharpe 0.70 (z_entry = 2.0, z_exit = 0.75, max_hold = 25), y hay configuraciones con 1 a 5 operaciones y Sharpe de hasta 0.59.
- Con 6 operaciones, una sola ganadora más o menos cambia el Sharpe en decenas de puntos: es tamaño de muestra, no estrategia.
- Los valores por defecto de Arturo dan 6 operaciones en train (Sharpe 0.33).
- **N_MIN no se bajó:** 675 configuraciones llegan a 30.

**Historia de la rejilla (declarada; todo con train y antes de evaluar test o valid):**
1. La primera rejilla (1,314 configuraciones) tenía el mejor en la orilla de max_hold_days (60) y de z_exit (0.25): la amplié a 150 días y a 0.1.
2. La elegida cayó en z_entry = 0.5, la orilla inferior: agregué 0.25 y la elección no cambió.
3. La primera regla de meseta (mediana ≥ 0.75 × Sharpe del mejor) no está definida con Sharpe negativo: la cambié por la del error estándar. Con ambas reglas la elección es la misma.

**Meseta o pico:**
- El vecindario de una configuración son las que difieren a lo más un paso por dimensión.
- El mejor es meseta si la mediana de su vecindario queda a lo más 1 error estándar del Sharpe por debajo del suyo. El error estándar es de Lo (2002), √((1 + SR²/2)/T) anualizado, ≈ 0.47 con 1,161 días.
- **El mejor bruto (0.75 / 0.25 / 3.0 / 150; Sharpe 0.12, 34 operaciones) es un pico aislado:** 51% de su vecindario no llega a 30 operaciones. Está al borde del acantilado de viabilidad.
- Por eso se toma el **centro de la mejor meseta**: la configuración viable con la mayor mediana de vecindario.
- La elegida es **z_entry = 0.5, z_exit = 0.25, z_stop = 3.5, max_hold_days = 150**: Sharpe de train −0.36 con rebalanceo diario y 47 operaciones. Su vecindario sí es meseta: 83% viable, mediana −0.34, Sharpe de −0.83 a 0.12.

**Orillas:**
- **max_hold_days = 150 está en la orilla.** En train la tenencia máxima no se activa: 0 salidas por tenencia y la operación más larga dura 145 días. El objetivo de train es idéntico para cualquier valor mayor, así que la superficie es plana más allá de la orilla y no es un óptimo interior.
- z_stop = 3.5 es el centro de un tramo plano: en train ningún |z| dentro de una operación pasa de 3, así que 3.0, 3.5, 4.0 e ∞ dan lo mismo.
- z_entry y z_exit son interiores. Por debajo de z_entry = 0.5 el Sharpe cae a −0.83.

**Banda de no-rebalanceo** (`band_grid.csv`, train, umbrales congelados):
- El Sharpe sube de −0.355 (diaria) a −0.321 con una banda de 5%. De 5% en adelante ya no hay rebalanceos en train y la superficie es plana hasta ∞.
- La regla (mayor Sharpe de train; en empate, la banda más chica) elige **5%**. No es una orilla de la rejilla, pero es el principio de un tramo plano.
- Política final = banda (mayor Sharpe de train).

## 5. Resultados por tramo

Parámetros congelados en train. Walk-forward = ancla fija: se calibra en train y se evalúa train → test → valid. Es un supuesto mío y queda como pregunta al profesor. **Valid se evaluó una sola vez.**

| | train | test | valid |
|---|---|---|---|
| Operaciones (largas / cortas) | 47 (33 / 14) | 37 (19 / 18) | 34 (19 / 15) |
| Sharpe | −0.32 | 0.38 | 0.03 |
| Sortino | −0.45 | 0.61 | 0.04 |
| Calmar | −0.12 | 0.30 | 0.01 |
| Drawdown máximo | −4.80% | −4.63% | −7.60% |
| Win rate | 51.1% | 70.3% | 64.7% |
| Profit factor | 0.49 | 1.72 | 1.02 |
| Rendimiento total / CAGR | −2.70% / −0.59% | +4.26% / +1.40% | +0.15% / +0.05% |
| Volatilidad anual | 1.8% | 3.8% | 4.0% |

**La estrategia pierde dinero en train y queda prácticamente en cero en valid.** No se ajustó nada después de ver test o valid. Lo explico en §6 y §7.

**Atribución del P&L bruto al neto:**

| | train | test | valid |
|---|---|---|---|
| P&L bruto | $70,559 | $122,240 | $71,561 |
| Comisiones | −$94,257 | −$77,727 | −$67,875 |
| Préstamo | −$3,299 | −$1,928 | −$2,169 |
| **P&L neto** | **−$26,997** | **+$42,585** | **+$1,517** |
| Comisiones / P&L bruto | 134% | 64% | 95% |

- El bruto es positivo en los tres tramos: 89%, 89% y 79% de las operaciones ganan en bruto. **La reversión existe, pero es chica frente al costo.**
- Cada viaje redondo cuesta 2 × 0.125% × 80% del capital ≈ $2,000 (más los rebalanceos), y la ganancia bruta media por operación es de $1,501 en train, $3,304 en test y $2,105 en valid.
- El préstamo es menor (≈ 2-4% del costo total): 0.25% anual sobre ≈ $400k en corto.

**Exposición media del capital:**

| | train | test | valid |
|---|---|---|---|
| Promedio de todos los días | 57.6% | 49.1% | 56.9% |
| Promedio de los días con posición | 78.7% | 78.5% | 78.4% |
| Fracción de días con posición | 73% | 63% | 73% |

- Con posición abierta la exposición queda debajo de 80% por tres razones:
  - la reserva para la comisión (−0.08 puntos);
  - el truncamiento a acciones enteras;
  - la deriva dentro de la banda de 5%, que solo puede bajar la exposición porque el tope de 80% obliga a rebalancear si sube.
- El promedio de todos los días es menor porque la estrategia está plana del 27% al 37% del tiempo, esperando |z| > 0.5.

**Rebalanceo diario contra banda** (mismas señales, mismos tramos; `rebalance_comparison.csv`):

| | train diaria → banda | test diaria → banda | valid diaria → banda |
|---|---|---|---|
| Comisiones | $99,068 → $94,257 (−$4,811) | $81,165 → $77,727 (−$3,438) | $71,961 → $67,875 (−$4,086) |
| Órdenes por pata | 5,249 → 1,649 | 2,992 → 1,018 | 3,372 → 924 |
| P&L bruto | $72,117 → $70,559 | $120,566 → $122,240 | $75,828 → $71,561 |
| P&L neto | −$30,287 → −$26,997 | $37,450 → $42,585 | $1,666 → $1,517 |
| Sharpe | −0.355 → −0.321 | 0.338 → 0.385 | 0.034 → 0.033 |

- La banda ahorra entre 4% y 6% de las comisiones y reduce las órdenes por pata entre 66% y 73%.
- El costo es seguir con menos precisión a β̂_t. En valid ese error de seguimiento costó $4,267 de P&L bruto, más que lo ahorrado, y el neto queda $149 peor.
- β̂_t cambia despacio, así que el rebalanceo diario aporta poco y cuesta comisiones.

## 6. Cómo pierde dinero la estrategia

**No pierde por muchas pérdidas pequeñas: pierde con unas pocas posiciones que dejan de revertir.**

- **train:** las 5 operaciones de más de 60 días suman −$31,663, más que todo el neto del tramo (−$26,997). La peor (2017-02-28 a 2017-09-25, 145 días) pierde $12,706. Las otras 42 operaciones suman +$4,666.
- **test:** dos operaciones largas (2020-04-23 a 2020-10-08, 117 días, y 2021-07-15 a 2022-01-10, 124 días) pierden $46,846. Las 3 peores suman −$51,196 de −$58,760 de pérdidas.
- **valid:** las 3 peores suman −$60,480 de −$79,002 de pérdidas.
- Los retornos por operación tienen colas pesadas: curtosis en exceso de 7.4 en train y test y de 9.6 en valid (figura 8).

**El stop sobre |z|.** Verifiqué el hallazgo de Arturo con mis propias cifras (`exit_analysis.json`). **Lo confirmo.**

| | train | test | valid |
|---|---|---|---|
| Días en que \|z\| > 3 (stop por defecto) | 0 | 0 | 2 |
| Días en que \|S_t − μ̂\| > 3 × σ̂ mediana de train | 0 | 140 | 177 |
| Salidas por señal / stop / tenencia, parámetros finales | 46 / 0 / 0 | 37 / 0 / 0 | 32 / 1 / 0 |
| Lo mismo con los valores por defecto (solo señales) | 4 / 0 / 2 | 1 / 0 / 2 | 1 / 0 / 2 |

- En test y valid el spread se aleja de su media más de 3 desviaciones "de train" en 140 y 177 días, pero |z| pasa de 3 solo en 2 días. La razón: σ̂ del filtro 2 crece con el choque y |z| regresa a la banda.
- **Episodio de marzo de 2024 (valid):**
  - El 2024-02-29, z = −3.90 (el único día del tramo con |z| > 3.5).
  - Desde el 2024-03-01, z nunca baja de −2.18, aunque el spread toca −0.2196 el 2024-03-06.
  - σ̂ pasa de 0.0296 (2024-02-28) a 0.1025, 3.5 veces más.
- Con mis parámetros, la posición larga abierta el 2024-02-27 **sí salió por stop el 2024-02-29**, con −$35,556, la peor operación de valid. Al día siguiente z = −2.18 reabrió largo, y esa operación ganó $26,783 en 69 días.
- Es decir, el stop sobre |z| solo actúa el primer día de un choque, antes de que σ̂ se adapte. Y en este caso vendió en el peor punto.

**La defensa real es la tenencia máxima, y con estos datos no ayuda:**
- En train, cortar las posiciones largas antes (max_hold de 31 a 60 días) da peor Sharpe que dejarlas correr. Con z_entry = 0.5 y z_exit = 0.25 el Sharpe es −0.66 con 31 días, −0.50 con 60 y −0.36 con 150. Las posiciones que se cortarían todavía no revierten y se cerrarían con pérdida.
- Por eso la calibración lleva la tenencia máxima a la orilla (150) y la estrategia queda sin defensa contra una ruptura de la relación.
- Un stop sobre el nivel del spread, y no sobre z, sería la defensa natural. No lo agregué porque cambia la regla de Arturo y requiere un acuerdo del equipo. Queda como pregunta.

## 7. Limitaciones

1. **Estabilidad del 79%:** 11 de 14 sub-ventanas, y las 3 inestables son la 10, la 13 y la 14, incluidas las 2 últimas de train (Milca). Es un aviso de ruptura justo antes de test.
2. **Deterioro en valid:** las innovaciones del filtro 2 tienen varianza 1.39 y Ljung-Box p = 0.027 (Arturo). Valid tiene el drawdown más profundo (−7.60%) y la peor operación (−$35,556).
3. **Test y COVID:** σ̂ promedio se duplica (0.0143 en train → 0.0300 en test). Con los valores por defecto casi no hay señales (3 operaciones en test). Con mis umbrales bajos hay 37, pero las dos operaciones de más de 100 días de 2020-2022 concentran las pérdidas.
4. **Número de operaciones:** 47, 37 y 34. Llegar a 30 en train obliga a z_entry = 0.5, una entrada a media desviación, donde la ganancia bruta por operación apenas cubre la comisión. Con 754 a 1,161 días por tramo, el error estándar del Sharpe anual es de 0.47 a 0.58: ningún Sharpe de la tabla de §5 se distingue de cero.
5. **Data snooping:**
   - **Deflated Sharpe Ratio** (Bailey y López de Prado, 2014; `metrics.json` → `deflated_sharpe`). La varianza del Sharpe entre pruebas sale de las 675 configuraciones viables de mi rejilla.
   - El umbral de Sharpe anual esperado por azar es 2.08 con las 910 pruebas de Johansen (45.5 falsos positivos esperados al 5%), 2.26 con las 2,535 configuraciones de mi rejilla, 2.32 con la suma y 3.24 con el producto (910 × 2,535).
   - El Sharpe de train es −0.32, así que **el DSR es ≈ 0 con cualquier conteo**.
   - Fuera de muestra (una prueba por tramo), la probabilidad de que el Sharpe verdadero sea > 0 (PSR) es 0.75 en test y 0.52 en valid.
6. Supuestos que favorecen a la estrategia: ejecución al cierre sin spread bid-ask ni impacto de mercado, y préstamo siempre disponible a 0.25%.

## 8. Conclusiones

- **La relación de equilibrio parece estructural en bruto, pero no es explotable neta de costos.**
  - A favor de que es estructural: el spread sigue revirtiendo fuera de muestra. El P&L bruto es positivo en test y valid, y 79% a 89% de las operaciones ganan en bruto, con parámetros congelados en train. La razón económica es la de Milca: monopolios regulados que se valúan casi como el mismo bono.
  - En contra, o al menos sin poder descartar el artefacto: el grupo salió de 910 pruebas, la traza no pasa Bonferroni y el DSR de train es ≈ 0. La señal además se debilita en valid (innovaciones peor calibradas, peor drawdown).
- **Neta de costos, la estrategia no tiene un rendimiento distinguible de cero en ningún tramo:** −2.70% en train, +4.26% en test y +0.15% en valid.
- La comisión de 0.125% por pata es del mismo orden que la reversión típica entre entrada y salida. Para que la estrategia valga la pena harían falta menos operaciones con más reversión cada una, que es justo lo que la restricción de operaciones mínimas no permite con 4.6 años operables de train.
