# Handoff · Milca · data, cointegración y filtro de Kalman 1

Rama: `milca/cointegracion-kalman1`. Para Arturo (filtro 2, VECM, señales) y Paula (sizing, backtest, dashboard).

## Cómo se corre

```
python main.py              # data -> cointegration -> kalman.filter1_weights (luego siguen las etapas de Arturo y Paula)
python -m pytest            # mis 4 archivos de prueba: 13 casos, todos pasan
```

- Python 3.12 con `requirements.txt`. Las descargas están congeladas en `data/raw/<sector>.csv`, así que no hace falta internet.
- Mis etapas tardan unos **6.5 minutos** en una Mac M2 de 8 núcleos (`cointegration` midió 379 s y 431 s de cómputo activo el 2026-10-30). `cointegration` evalúa 910 grupos de 8 sectores, en paralelo con un proceso por núcleo, y el resultado no depende del número de núcleos. Dos corridas desde cero dieron artefactos idénticos byte por byte.
- **Tiempo esperado de `cointegration`:**
  - **Mac M2 (8 núcleos): unos 6.5 minutos.**
  - **Windows 11 (8 núcleos, máquina de Arturo): unos 50 minutos** (3,006 s).
  - El 90% del tiempo es la rejilla de q de cada grupo: el filtro de Kalman y las pruebas ADF, que son regresiones de álgebra lineal (BLAS).
  - **Causa probable de la diferencia, todavía sin comprobar en Windows:** sobresuscripción de hilos.
    - En Mac, numpy usa Accelerate de Apple.
    - En Windows, numpy usa OpenBLAS, que abre por su cuenta un hilo por núcleo. Con 8 procesos en paralelo quedan unos 64 hilos peleando por 8 núcleos, y los hilos de OpenBLAS esperan activamente, gastando CPU sin avanzar.
    - El arranque "spawn" de Windows (cada proceso hijo vuelve a importar las librerías, en 9 grupos de procesos) suma segundos, no decenas de minutos.
  - Si la computadora se suspende a media corrida (laptop con batería o tapa cerrada), el tiempo de reloj se dispara aunque el cálculo no cambie. Corran `python main.py` conectados a la corriente y sin suspender.
**Pendiente para Arturo (estructura del repo; yo no toco esos archivos):**
1. Crear `pytest.ini` en la raíz para que también funcione `pytest` a secas, con exactamente:
   ```
   [pytest]
   pythonpath = .
   testpaths = tests
   ```
2. Agregar `yfinance==1.7.0` a `requirements.txt`. Solo hace falta para volver a descargar datos; el pipeline no lo necesita mientras existan los archivos congelados.

## Grupo final y por qué

**Grupo final: eléctricas reguladas DUK, XEL, ED, ES, CMS, AEE (6 activos).** Todo se eligió con train (2011-01-03 a 2019-12-30).

- En v4 (oro) no se alcanzaba la correlación de 0.7 y la estabilidad era del 21%.
- Antes de descargar nada se registraron en git (commit `e52f6a2`) 7 sectores nuevos, sus candidatos con justificación por activo y una regla de selección con 7 requisitos:
  - correlación de retornos > 0.7;
  - r̂ ≥ 1 por traza y por máximo eigenvalor;
  - ADF del spread;
  - vida media de 5 a 60 días;
  - meseta de q con cero cambios de signo;
  - ADF del spread de Kalman;
  - estabilidad ≥ 50%.
- Se evaluaron 910 grupos. **4 cumplen los 7 requisitos, todos de eléctricas.** El ganador es el de mayor estabilidad: **79%** (11 de 14 sub-ventanas).
- La razón económica: son monopolios regulados cuyo rendimiento fija el regulador, así que se valúan casi como el mismo bono de largo plazo.

| Pata | β (Johansen, train) | \|β\| | Participación en el nocional |
|---|---|---|---|
| **XEL** (normalización) | **+1** | 1.000 | 36.0% |
| ES | −0.658 | 0.658 | 23.7% |
| ED | −0.484 | 0.484 | 17.4% |
| AEE | −0.321 | 0.321 | 11.5% |
| DUK | +0.186 | 0.186 | 6.7% |
| CMS | +0.131 | 0.131 | **4.7%** (debajo del 5%; ver abajo) |

## Dirección de las posiciones (leer con cuidado)

**Estar largo el spread significa comprar cada activo con β_i > 0 y vender cada activo con β_i < 0, en proporción a |β_i|. Estar corto el spread invierte todos los signos. El nocional se reparte entre todos los activos en proporción a |β_i|.**

- **Pata de normalización:** XEL, que vale +1 en todas las fechas de `beta_t.csv`.
- **Convención de signo:** la regla de arriba. Con el β de Johansen, largo el spread = comprar XEL, DUK y CMS y vender ES, ED y AEE; los montos son proporcionales a |β| por pata: XEL 1.000, ES 0.658, ED 0.484, AEE 0.321, DUK 0.186, CMS 0.131.
- **Son log-precios:** cada peso es **exposición en valor**, no número de acciones. El nocional en dólares de cada pata es proporcional a |β_i|, con el signo de β_i.
- **Usen los pesos dinámicos β̂_t de `artifacts/kalman/beta_t.csv`**, no el β estático. Con β̂_t, durante train después del burn-in:
  - largo el spread = comprar XEL y DUK;
  - vender ED, ES, CMS y AEE.
- **Ojo con CMS:** en Johansen vale +0.131, pero en el filtro vale alrededor de −0.29 en todo train después del burn-in (sin cambiar de signo). Con una pata tan pequeña el signo no está bien identificado. Por eso deben usar β̂_t del día.

> **Advertencia:** un error de signo produce una estrategia que pierde de forma sostenida y **parece mala suerte**. Verifiquen la dirección sobre un episodio conocido de train. Cuando el spread esté muy arriba de su media, la posición debe ser corta en el spread y debe ganar cuando el spread baje. Por ejemplo, del 2014-05-07 al 2014-05-13 el spread es positivo y crece: XEL está cara frente a la canasta.

**Vida media del spread (train):** **15.3 días** (Johansen) y **20.4 días** (Kalman, después del burn-in). Úsenla para calibrar la tenencia máxima.

**Serie que deben usar como spread:** la columna `spread` de `artifacts/kalman/spread_t.csv`.
- Es S_t = β̂_tᵀ(log P_t − μ_train), calculada con los pesos previos (sin look-ahead), e igual a la innovación ỹ_t.
- Al lado vienen `innovation_std` (√S_t) e `innovation_normalized`.
- Las filas con `burn_in = True` (hasta 2014-05-07) son del arranque del filtro: no las usen para calibrar.

## Cifras clave (todas de train: 2011-01-03 a 2019-12-30)

**Búsqueda:**
- 8 sectores, 73 activos y 938 subgrupos de 6, más 37 recortes: **975 grupos** (17,162 contando versiones anteriores).
- 910 pruebas de Johansen en train. Se esperan 45.5 falsos positivos por azar al 5%.
- Respaldo de correlación en 4 sectores (no afecta al ganador).

**Johansen (ganador):**
- p = 1 (AIC y BIC); constante sin restringir (caso 3); valores críticos de MacKinnon, Haug y Michelis.
- **r̂ = 1, con traza y máximo eigenvalor de acuerdo:**
  - traza 101.56 contra 95.75 (no pasa al 99%: 104.96);
  - máximo eigenvalor 51.57 contra 40.08 (sí pasa al 99%: 45.87).
- Eigenvalores: 0.0225, 0.0117, 0.0050, 0.0039, 0.0013, 0.0002.

**Spread de Johansen:** ADF −6.08, p = 1.1·10⁻⁷; vida media de 15.3 días.

**Correlación de retornos por sub-ventana:** 0.742, 0.840 y 0.838.

**Estabilidad:** 11 de 14 sub-ventanas (79%). Las 3 inestables son las últimas de train: posible ruptura posterior.

**Filtro 1:**
- **q = 1.12·10⁻⁵**, el centro de una **meseta de 15 q contiguos** (5.0·10⁻⁶ a 2.5·10⁻⁵) en una rejilla fina de 141 valores. Cumplen 25 de 141, y no está en la orilla.
- **Cambios de signo en train: 0.**
- R = 2.81·10⁻⁴ (√R = 0.0168).
- Diagonal de Q: de 1.7·10⁻⁸ a 5.1·10⁻⁸.
- P₀ = 10⁴·I; **burn-in de 840 días (37.1% de train)**, hasta 2014-05-07.
- Piso de la varianza posterior: de 1.1·10⁻⁴ a 4.2·10⁻⁴.
- **Robustez a P₀:** los pesos difieren como máximo 8.3·10⁻⁶.
- Spread del filtro: ADF p = 0.0136; vida media de 20.4 días.

**Respaldos:** 4 en la etapa 2 (sectores que no pasaron la correlación); 0 en q.

## Archivos que produzco

| Archivo | Contenido |
|---|---|
| `data/universe.json` | Preregistro v5: 8 sectores, 77 candidatos con justificación, protocolo y regla |
| `data/raw/<sector>.csv`, `<sector>_meta.json` | Descargas congeladas |
| `data/processed/prices.csv` | **Todos los candidatos (73 activos, USD)** + `period`. **El grupo bueno está en `selected_group.json` → `tickers`**; tomen solo esas 6 columnas |
| `data/processed/sectors.json` | Tickers de cada sector y excluidos |
| `data/processed/splits.json`, `cleaning_report.json` | Partición y conteos de validación |
| `artifacts/cointegration/group_selection.csv` | Los 938 subgrupos del embudo por sector |
| `artifacts/cointegration/candidatos_final.csv` | Los 910 grupos evaluados con R1-R7 y su orden |
| `artifacts/cointegration/q_grids_candidatos.csv` | Rejilla de q de cada candidato |
| `artifacts/cointegration/selected_group.json` | Grupo ganador: tickers, r̂, estadísticos y valores críticos, columnas de β, vector elegido, pata, signo, ADF, vida media, estabilidad, participación en el nocional y conteo de pruebas |
| `artifacts/cointegration/stability_windows.csv` | Etapa 7 del ganador |
| `artifacts/cointegration/eigvec_rolling.csv` | Primer eigenvector móvil (solo datos pasados) + `period` |
| `artifacts/kalman/beta_t.csv` | β̂_t normalizado (XEL = +1) + `period`, `burn_in` |
| `artifacts/kalman/spread_t.csv` | `spread`, `innovation_normalized`, `innovation_std`, `period`, `burn_in` |
| `artifacts/kalman/gain_1.csv`, `posvar_1.csv` | K_t y diagonal de P_{t\|t} |
| `artifacts/kalman/filter1_q_grid.csv`, `filter1_params.json` | Rejilla de q del ganador y parámetros finales |
| `artifacts/figures/` | Figuras 1, 3 y 4, diagnóstico de convergencia y rejilla de q |

Todas las salidas con fecha tienen la fecha ISO en la primera columna, columnas en minúsculas y la columna `period`.

## Aclaración sobre test y valid

El filtro y el eigenvector móvil se **corren** hacia adelante en toda la muestra con parámetros **congelados en train**, porque ustedes necesitan esas series. Nada se estima, calibra ni elige con test o valid, y ninguna cifra de esos tramos se usó para decidir. No es look-ahead.

## Supuestos

- Los precios ajustados de Yahoo son correctos. Se marcaron 2 saltos, ninguno parece error.
- La constante es un término determinístico (la media de train), no un estado.
- Train es "en muestra" para el filtro.

## Límites conocidos

1. **Pruebas múltiples:**
   - 975 grupos en v5 (17,162 en total) y 910 pruebas de Johansen.
   - La traza del ganador no pasa al 99%, así que no sobrevive a Bonferroni.
   - Lo respaldan los otros 6 requisitos y que 21 de los 23 mejores grupos son eléctricas.
   - El backtest debe reportar el Deflated Sharpe Ratio o el Reality Check.
2. Burn-in del 37% de train.
3. Las 3 sub-ventanas inestables están al final de train: posible pérdida de cointegración en test o valid.
4. CMS pesa 4.7% y su signo en Kalman difiere del de Johansen.
5. Las innovaciones del filtro 1 están autocorrelacionadas (0.967), porque son el spread que revierte. La prueba de innovaciones blancas aplica al filtro 2.
6. **Tiempo en Windows:** `cointegration` tarda unos 50 minutos en Windows contra 6.5 en Mac. Para comprobar la causa sin tocar código, se puede correr en PowerShell `$env:OPENBLAS_NUM_THREADS="1"; python main.py`, que deja un hilo de BLAS por proceso, y revisar con `git diff --stat artifacts/` que las cifras no cambian más que en ruido de punto flotante. Hay dos opciones para acelerarlo sin cambiar resultados, todavía sin aplicar: fijar un hilo de BLAS por proceso dentro del código, o reusar la etapa si sus entradas no cambiaron.
7. **Codificación:** todas mis lecturas y escrituras de texto usan `encoding="utf-8"` y lo que imprimo es ASCII. Se verificó con `python -X warn_default_encoding` y con la consola forzada a cp1252.
