# Lab 03 · Arbitraje estadístico con cointegración y filtros de Kalman

Microestructuras y Sistemas de Trading (MyST) · ITESO · Equipo 2

## Integrantes

- Paula Inés Pelayo Morales
- Milca Lilian Correa Sánchez
- Arturo Santillanes Llamas

## Nivel de alcance

Se desarrolla todo el enunciado salvo la sección 3.7 (Risk Parity entre varios grupos): un grupo de 6 activos cointegrados. El tamaño del grupo se justifica como 2·n, con n = 3 integrantes.

## Descripción del proyecto

Estrategia long/short de reversión a la media sobre un grupo de 6 activos que comparten una tendencia estocástica común. El grupo se elige usando solo la partición de entrenamiento (correlación, Engle-Granger y Johansen). Johansen da el vector de pesos β; el filtro de Kalman 1 estima pesos dinámicos β̂_t y el spread S_t = β̂_tᵀP_t; el filtro de Kalman 2 estima la media y la varianza latentes del spread para calcular un z-score sin look-ahead (con estadísticos previos a la actualización). Con un VECM y señales por z-score (entrada, salida, stop y tenencia máxima) se opera con 80% del capital repartido según los pesos, con costos reales (comisión 0.125% por pata y préstamo 0.25% anual) y capital inicial de $1,000,000. Los datos son diarios de 10 a 15 años, con partición cronológica 60/20/20 (train/test/valid). Incluye dashboard, pruebas y reporte.

## Estructura del repositorio

```
Lab03_MyST_Equipo2/
├── README.md
├── requirements.txt          # dependencias con versiones fijadas
├── .gitignore
├── main.py                   # pipeline completo, un solo comando
├── data/                     # ingesta, limpieza y validación
├── cointegration/            # screening, Engle-Granger, Johansen
├── kalman/                   # ambos filtros
├── strategy/                 # VECM, z-score, señales, sizing
├── backtest/                 # motor con costos y walk-forward
├── performance/              # cálculo de métricas
├── dashboard/
│   └── app.py                # Streamlit
├── project_management/
│   ├── charter.md
│   ├── sprint_01.md
│   ├── sprint_02.md
│   ├── postmortem.md
│   ├── contrato.md           # archivos que se pasan entre módulos
│   └── ia_log.md             # registro de uso de IA
├── artifacts/                # salidas del backtest que lee el dashboard
├── tests/
└── docs/
    └── reporte.pdf
```

## Instalación

Verificado en un clon limpio con Python 3.14.7.

1. Clonar el repositorio: `git clone https://github.com/ArturoSantillanes/Lab03_MyST_Equipo2.git` y entrar a la carpeta.
2. Crear un entorno virtual: `python -m venv .venv`
3. Activarlo: en Windows `.venv\Scripts\activate`; en macOS/Linux `source .venv/bin/activate`.
4. Instalar dependencias: `pip install -r requirements.txt`

## Cómo reproducir (comando exacto)

```
python main.py
```

Ejecuta en orden: data → cointegration → kalman.filter1_weights → kalman.filter2_spread → strategy → backtest → performance. Imprime cada etapa con su duración y se detiene en el primer error indicando qué etapa falló.

Tiempo medido en Windows 11 con 8 núcleos: de 42 a 44 minutos en total, de los cuales la etapa `cointegration` toma de 37 a 39. En una Mac M2, esa etapa tarda unos 6.5 minutos.

## Dashboard

```
streamlit run dashboard/app.py
```

El dashboard solo lee los archivos de `artifacts/`; no vuelve a correr el backtest. Por eso primero hay que correr `python main.py`.

### Guía de uso del dashboard (una página)

Se abre con `streamlit run dashboard/app.py` (después de `python main.py`) en http://localhost:8501. Solo lee `artifacts/`: abrirlo no vuelve a correr el backtest.

**Barra lateral (filtros que afectan a todas las gráficas)**
- **Tramos:** marca o quita train, test y valid.
- **Rango de fechas:** recorta todas las series de tiempo.
- Debajo aparecen el grupo, los umbrales finales (calibrados solo en train) y la política de rebalanceo.

**Pestaña 1 · Vista ejecutiva**
1. *Curva de capital y drawdown:* equity neto de costos de cada tramo (cada uno empieza con $1,000,000) y su drawdown debajo. Las líneas punteadas son las fronteras entre tramos.
2. *Tarjeta de métricas:* Sharpe, Sortino, Calmar, drawdown máximo, win rate y profit factor de cada tramo, más fechas, número de operaciones y rendimiento total.
3. *Atribución de costos:* cascada del P&L bruto al neto (comisiones y préstamo) por tramo.
4. *Bitácora de operaciones:* tabla filtrable por dirección (largo/corto) y resultado (ganadora/perdedora), además de los filtros de la barra lateral. Se ordena con clic en el encabezado y se descarga como CSV desde el menú de la tabla.
5. *Distribución de retornos:* histograma de los retornos netos por operación (filtrados) con una normal superpuesta.

**Pestaña 2 · Vista técnica**
1. Spread S_t con la media μ̂ del filtro 2, entradas (triángulos) y salidas (marcadas por motivo).
2. Z-score con las líneas ±z_entry, ±z_exit y ±z_stop.
3. β̂_t de cada activo (filtro 1).
4. Primer eigenvector de Johansen móvil.
5. Ganancia de Kalman de ambos filtros: filtro 1 por activo, sin xel, que es la pata de normalización; filtro 2 para μ y σ².
6. Varianza posterior de ambos filtros (escala log).
7. Sharpe móvil de 252 días.

En las gráficas de los filtros el eje se ajusta al tramo sin burn-in; el arranque se ve con el zoom o con doble clic.

**Pestaña 3 · Calibración**
- Mapa de calor del Sharpe de train por z_entry × z_exit, con el número de operaciones en cada celda; z_stop y max_hold se eligen arriba. Sirve para ver la restricción de 30 operaciones y la meseta.
- Comparación de rebalanceo diario contra banda y Sharpe de train por banda.
- Deflated Sharpe Ratio, PSR fuera de muestra y cuántas veces se evaluó valid.

**Uso de las gráficas (Plotly)**
- Arrastra para hacer zoom y haz doble clic para volver.
- Pasa el cursor para ver los valores del día.
- Haz clic en la leyenda para ocultar o mostrar una serie, o doble clic para aislarla.
- El ícono de cámara descarga la gráfica como PNG.

Si falta algún archivo de `artifacts/`, el dashboard lo dice y pide correr `python main.py`.

## Semilla

Semilla aleatoria **42**, fijada en la constante `SEED` de [main.py](main.py) para `random` y `numpy`.

## Pruebas

```
pytest
```

## Uso de asistencia de IA

Resumen por persona; el detalle (qué generó la IA y cómo se verificó) está en [project_management/ia_log.md](project_management/ia_log.md).

- **Milca:** usó IA (Claude Code) en `data/` (incluye el preregistro de sectores `data/universe.json`), `cointegration/` (embudo de 7 etapas por sector, selección final con los requisitos R1-R7 y figuras 1, 3, 4, de diagnóstico y de la rejilla de q), `kalman/filter1_weights.py`, sus pruebas (`test_split`, `test_cointegration`, `test_beta_stability`, `test_filter1_step`), `project_management/handoff_milca.md` y `docs/secciones/milca.md`; revisó, probó y entendió su código.
- **Arturo:** usó IA (Claude Code) en el armado del repositorio y `project_management/` (incluye `pytest.ini`, charter, sprints y `handoff_arturo.md`), `kalman/filter2_spread.py` (filtro 2), `strategy/` (`vecm.py`, `zscore.py`, `signals.py`, `figures.py` con las figuras 2, 5 y 6), sus pruebas (`test_causality`, `test_innovations`, `test_signals`, `test_filter2_step`), `docs/secciones/arturo.md`, la auditoría final, el armado de `docs/reporte.pdf`, el borrador de `postmortem.md` y `sprint_02.md`; revisó, probó y entendió su código.
- **Paula:** usó IA (Claude Code) en `strategy/sizing.py`, `backtest/` (`engine.py`, `costs.py`, `calibration.py`, `walk_forward.py`, `io.py` y la etapa `backtest`), `performance/` (`metrics.py`, `rolling.py`, `figures.py` con las figuras 7 y 8, y la etapa `performance`), `dashboard/app.py`, sus pruebas (`test_accounting`, `test_metrics`, `test_exposure_cap`), `project_management/handoff_paula.md` y `docs/secciones/paula.md`; revisó, probó y entendió su código.
