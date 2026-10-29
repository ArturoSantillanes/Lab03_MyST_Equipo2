# Handoff · Milca · data, cointegración y filtro de Kalman 1

Rama: `milca/cointegracion-kalman1`. Para Arturo (filtro 2, VECM, señales) y Paula (sizing, backtest, dashboard).

## Cómo se corre

```
python main.py              # data -> cointegration -> kalman.filter1_weights (luego siguen las etapas de Arturo y Paula)
python -m pytest            # mis 4 pruebas: 11 casos, todos pasan
```

- Python 3.12 con `requirements.txt`. La descarga ya está congelada en `data/raw/prices_raw.csv`, así que no hace falta internet.
- La etapa `cointegration` tarda alrededor de 1 minuto: calcula el eigenvector móvil en cada fecha y evalúa los candidatos finales.

**Pendiente para Arturo (estructura del repo).** Para que también funcione `pytest` a secas, crear `pytest.ini` en la raíz con exactamente:

```
[pytest]
pythonpath = .
testpaths = tests
```

Para volver a descargar datos haría falta agregar `yfinance==1.7.0` a `requirements.txt`. El pipeline no lo necesita mientras exista la descarga congelada.

## Grupo final y por qué cambió

- **Grupo final: NEM, GFI, FNV, RGLD (4 activos).**
- El embudo de 7 etapas eligió primero 6 activos (NEM, KGC, GFI, HMY, FNV, RGLD).
- En ese grupo, **KGC y HMY tenían pesos casi nulos**: 1.6% y 2.2% del nocional, por debajo del umbral de relevancia del 5% fijado de antemano. Esos pesos oscilaban alrededor de cero y cambiaban de signo por ruido, así que `test_beta_stability` fallaba.
- Se evaluaron 4 candidatos con una regla escrita antes de ver la tabla (r̂ ≥ 1 con las dos pruebas, ADF, vida media de 5 a 60 días, cero cambios de signo; luego estabilidad). **El único elegible fue el de 4 activos**, que además es el más estable.
- El enunciado permite grupos de 2 a 8 activos. No se cambió ningún umbral.

| Pata | β (train) | \|β\| | Participación en el nocional |
|---|---|---|---|
| **FNV** (normalización) | **+1** | 1.000 | 40.0% |
| RGLD | −0.577 | 0.577 | 23.1% |
| GFI | +0.512 | 0.512 | 20.5% |
| NEM | −0.411 | 0.411 | 16.4% |

## Dirección de las posiciones (leer con cuidado)

**Estar largo el spread significa comprar cada activo con β_i > 0 y vender cada activo con β_i < 0, en proporción a |β_i|. Estar corto el spread invierte todos los signos. El nocional se reparte entre todos los activos en proporción a |β_i|.**

- Con el β de train: **largo el spread = comprar FNV y GFI; vender RGLD y NEM.**
- **Pata de normalización:** FNV (+1).
- **Convención de signo:** la de arriba. En `beta_t.csv`, FNV vale +1 en todas las fechas.
- **Son log-precios:** cada peso es **exposición en valor**, no número de acciones. El nocional en dólares de cada pata es proporcional a |β_i|, con el signo de β_i.

> **Advertencia:** un error de signo produce una estrategia que pierde de forma sostenida y **parece mala suerte**. Verifiquen la dirección sobre un episodio conocido de train: cuando el spread esté muy por arriba de su media, la posición debe ser corta en el spread y debe ganar cuando el spread baje.

**Vida media del spread:** **57.9 días** (Johansen) y **55.5 días** (Kalman), ambas en train. Úsenla para calibrar el periodo máximo de tenencia.

**Serie que deben usar como spread:** la columna `spread` de `artifacts/kalman/spread_t.csv`.
- Es S_t = β̂_tᵀ(log P_t − μ_train), con los pesos previos del filtro, e igual a la innovación ỹ_t.
- `innovation_std` (√S_t) e `innovation_normalized` vienen al lado.

## Cifras clave (todas de train: 2011-01-03 a 2019-12-30)

**Embudo:** 84 subgrupos; 0 pasan la correlación (máximo 0.652), así que se usó un **respaldo**: priorizar los 84. Llegaron 30 a Johansen y el embudo eligió el de 6.

**Selección final:** 4 candidatos, 1 elegible.

**Johansen (ganador):**
- p = 1 (AIC y BIC); constante sin restringir (caso 3); valores críticos de MacKinnon, Haug y Michelis (1996).
- **r̂ = 1, con traza y máximo eigenvalor de acuerdo:** traza 49.92 contra 47.85 (no pasa al 99%: 54.68); máximo eigenvalor 34.75 contra 27.59.
- Eigenvalores: 0.01525, 0.00357, 0.00293, 0.00020.

**Spread:** ADF −3.467, p = 0.0089; vida media de 57.9 días.

**Estabilidad:** 3 de 14 sub-ventanas (21%), ninguna después de 2015.

**Filtro 1:**
- q = 1.0·10⁻⁶ (rejilla de 29 valores; cumplen 2; no está en la orilla; no hay meseta: los dos q válidos no son contiguos).
- **Cambios de signo en train: 0** (NEM 0, GFI 0, RGLD 0).
- R = 0.01456 (√R = 0.121).
- Diagonal de Q: NEM 1.42·10⁻⁷, GFI 5.71·10⁻⁸, RGLD 1.83·10⁻⁷.
- P₀ = 10⁴·I; **burn-in de 908 días (40.1% de train)**, hasta 2014-08-13.
- Piso de P: de 1.8·10⁻⁴ a 4.6·10⁻⁴.
- **Robustez a P₀:** los pesos difieren como máximo 3.9·10⁻⁶.
- Spread del filtro: ADF p = **0.0476** (apenas pasa) y vida media de 55.5 días.

**Respaldos usados: 1** (etapa 2). En q: 0.

## Archivos que produzco

| Archivo | Contenido |
|---|---|
| `data/universe.json` | Universo de 9 activos del oro y su justificación |
| `data/raw/prices_raw.csv`, `download_meta.json` | Descarga congelada |
| `data/processed/prices.csv` | **Los 9 activos del universo** (USD) + `period`. **El grupo bueno está en `selected_group.json` → `tickers`**; tomen solo esas 4 columnas |
| `data/processed/splits.json`, `cleaning_report.json` | Partición y conteos de validación |
| `artifacts/cointegration/group_selection.csv` | Los 84 subgrupos del embudo (incluye la correlación en niveles solo como dato informativo) |
| `artifacts/cointegration/candidatos_final.csv` | Los 4 candidatos de la selección final, con todos sus estadísticos |
| `artifacts/cointegration/selected_group.json` | Grupo ganador: tickers, r̂, eigenvalores, estadísticos y valores críticos, columnas de β, columna elegida, pata, signo, ADF, vida media, orden p, especificación, estabilidad y participación en el nocional |
| `artifacts/cointegration/stability_windows.csv` | Etapa 7 del ganador |
| `artifacts/cointegration/eigvec_rolling.csv` | Primer eigenvector móvil (solo datos pasados) + `period` |
| `artifacts/kalman/beta_t.csv` | β̂_t normalizado (FNV = +1) + `period`, `burn_in` |
| `artifacts/kalman/spread_t.csv` | `spread`, `innovation_normalized`, `innovation_std`, `period`, `burn_in` |
| `artifacts/kalman/gain_1.csv`, `posvar_1.csv` | K_t y diagonal de P_{t\|t} |
| `artifacts/kalman/filter1_q_grid.csv`, `filter1_params.json` | Rejilla de q y parámetros finales |
| `artifacts/figures/` | Figuras 1, 3, 4 y de convergencia |

Todas las salidas de `artifacts/kalman/` tienen la fecha ISO en la primera columna, columnas en minúsculas y la columna `period`.

## Aclaración sobre test y valid

El filtro y el eigenvector móvil se **corren** hacia adelante en toda la muestra con parámetros **congelados en train**, porque ustedes necesitan esas series. Nada se estima, calibra ni elige con test o valid, y ninguna cifra de esos tramos se usó para decidir. No es look-ahead.

## Supuestos

- Los precios ajustados de Yahoo son correctos (0 errores encontrados).
- La constante es un término determinístico (la media de train), no un estado.
- Train es "en muestra" para el filtro.

## Límites conocidos

1. Evidencia de cointegración débil:
   - 1 de 36 pares pasa Engle-Granger (se esperaban 1.8 por azar).
   - 33 pruebas de Johansen (se esperan 1.65 falsos positivos).
   - El ganador no sobrevive a Bonferroni.
2. La etapa 2 de correlación no se cumple (0 de 84); el respaldo se adoptó después de ver el fracaso y está declarado como desviación.
3. Data snooping:
   - Se probaron dos sectores, y el oro se eligió después de que fallara energía.
   - 171 grupos evaluados en esta rama, 16,187 contando versiones anteriores.
4. Estabilidad del 21% y burn-in del 40% de train.
5. El ADF del spread de Kalman está en el límite (p = 0.0476), y el q elegido no viene de una meseta.
