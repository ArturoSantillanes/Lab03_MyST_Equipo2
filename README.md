# Lab 03 · Arbitraje estadístico con cointegración y filtros de Kalman

Microestructuras y Sistemas de Trading (MyST) · ITESO · Equipo 2

## Integrantes

- Paula Inés Pelayo Morales
- Milca Lilian Correa Sánchez
- Arturo Santillanes Llamas

## Nivel de alcance

**Nivel A:** un grupo de 6 activos cointegrados. El tamaño del grupo se justifica como 2·n, con n = 3 integrantes.

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
│   ├── postmortem.md
│   ├── contrato.md           # archivos que se pasan entre módulos
│   └── ia_log.md             # registro de uso de IA
├── artifacts/                # salidas del backtest que lee el dashboard
├── tests/
└── docs/
    └── reporte.pdf
```

## Instalación

1. Clonar el repositorio: `git clone https://github.com/ArturoSantillanes/Lab03_MyST_Equipo2.git` y entrar a la carpeta.
2. Crear un entorno virtual: `python -m venv .venv`
3. Activarlo: en Windows `.venv\Scripts\activate`; en macOS/Linux `source .venv/bin/activate`.
4. Instalar dependencias: `pip install -r requirements.txt`

## Cómo reproducir (comando exacto)

```
python main.py
```

Ejecuta en orden: data → cointegration → kalman.filter1_weights → kalman.filter2_spread → strategy → backtest → performance. Imprime cada etapa con su duración y se detiene en el primer error indicando qué etapa falló.

## Dashboard

```
streamlit run dashboard/app.py
```

El dashboard solo lee los archivos de `artifacts/`; no vuelve a correr el backtest. Por eso primero hay que correr `python main.py`.

### Guía de uso del dashboard (una página)

Pendiente: la completa Paula.

## Semilla

Semilla aleatoria **42**, fijada en la constante `SEED` de [main.py](main.py) para `random` y `numpy`.

## Pruebas

```
pytest
```

## Uso de asistencia de IA

Resumen por persona; el detalle (qué generó la IA y cómo se verificó) está en [project_management/ia_log.md](project_management/ia_log.md).

- **Milca:** usó IA (Claude Code) en `data/` (incluye `data/universe.json`), `cointegration/` (embudo de 7 etapas y figuras 1, 3, 4 y de diagnóstico), `kalman/filter1_weights.py`, sus pruebas (`test_split`, `test_cointegration`, `test_beta_stability`, `test_filter1_step`), `project_management/handoff_milca.md` y `docs/secciones/milca.md`; revisó, probó y entendió su código.
- **Arturo:** usó IA en el armado del repositorio, `project_management/`, `kalman/filter2_spread.py` y `strategy/` (vecm, zscore, signals); revisó, probó y entendió su código.
- **Paula:** usó IA en `strategy/sizing.py`, `backtest/`, `performance/` y `dashboard/`; revisó, probó y entendió su código.
