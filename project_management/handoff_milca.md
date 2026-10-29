# Handoff · Milca · data, cointegración y filtro de Kalman 1

Rama: `milca/cointegracion-kalman1`. Para Arturo (filtro 2, VECM, señales) y Paula (sizing, backtest, dashboard).

## Cómo se corre

```
python main.py              # data -> cointegration -> kalman.filter1_weights (luego siguen las etapas de Arturo y Paula)
python -m pytest            # mis 4 pruebas (ver "Límites": `pytest` a secas necesita un pytest.ini en la raíz)
```

- Python 3.12 con `requirements.txt`.
- La descarga ya está congelada en `data/raw/prices_raw.csv`, así que el pipeline **no** necesita internet ni yfinance. Para volver a descargar haría falta `yfinance==1.7.0`, que no está en `requirements.txt`.
- La etapa `cointegration` tarda alrededor de 1 minuto porque calcula el eigenvector móvil en cada fecha.

## Qué hice y qué decidí (todo con train: 2011-01-03 a 2019-12-30)

| Tema | Decisión | Por qué |
|---|---|---|
| Universo | 9 empresas de la cadena del oro (`data/universe.json`): NEM, AEM, KGC, AU, GFI, HMY, FNV, WPM, RGLD | Tendencia estocástica común = precio del oro; mineras grandes, medianas, africanas y empresas de regalías y *streaming* |
| Ventana | 2011-01-01 a 2025-12-31 (3,772 días) | 15 años, el máximo; train de ~9 años |
| Limpieza | 0 faltantes, 0 duplicados, 0 desalineadas, 0 saltos anómalos | Reglas en `data/clean.py` |
| Log-precios | En todo el bloque | Formulación del profesor |
| Etapa 2 | Correlación de log-retornos, ventana de 63 días, 3 sub-ventanas, media > 0.7 en todas | **0 de 84 pasan** (máximo 0.652). **Respaldo** (1 vez): la etapa 3 prioriza los 84 |
| Etapa 3 | Engle-Granger en los 15 pares; prioridad = pares que pasan y luego p medio | Solo 1 par distinto de 36 pasa (KGC-HMY) |
| Etapas 4 a 6 | Secuencial por prioridad; Johansen con p por BIC (= 1, AIC también 1); constante sin restringir | **30 subgrupos llegaron a Johansen**; el nº 30 cumple |
| Etapa 7 | 14 sub-ventanas de 504 días cada 126 | Cointegrado en **2 de 14 (14%)** |
| Filtro 1 | z_t = β_t sin intercepto, A = I, log-precios centrados con la media de train | Notación del profesor; consistente con la constante sin restringir de Johansen |
| q | Rejilla de 29 valores (10⁻⁸ a 10⁻¹); criterio fijado antes: (a) ADF, (b) vida media de 5 a 60 días, (c) sin cambios de signo, (d) pesos que se adaptan | **Ningún q cumple (c).** **Respaldo** (1 vez): el de menos cambios de signo entre los que cumplen (a), (b) y (d) |
| R | Varianza residual OLS estática en train | Ruido que la canasta no explica |
| P₀ | 10⁴·I (difuso) | Los \|β\| normalizados son ≤ 1 |
| Burn-in | Medido: P de todos los componentes bajo el techo de su banda (mediana ± 3 MAD del log de P) hasta el final de train | Salió de 842 días |

## Cifras clave (train)

**Grupo final:** **NEM, KGC, GFI, HMY, FNV, RGLD**.

**Johansen:**
- Orden p: AIC = 1, BIC = 1; se usa BIC.
- Especificación: constante sin restringir (caso 3), fuera de la relación; valores críticos de MacKinnon, Haug y Michelis (1996).
- **r̂ = 1**, con traza y máximo eigenvalor de acuerdo: traza 99.75 contra 95.75, máximo eigenvalor 45.96 contra 40.08.
- Eigenvalores: 0.02011, 0.01003, 0.00680, 0.00456, 0.00219, 0.00011.

**Vector y spread:**

| | NEM | KGC | GFI | HMY | FNV | RGLD |
|---|---|---|---|---|---|---|
| β (columna 0, normalizado) | −0.314 | +0.041 | +0.474 | −0.057 | **+1** | −0.741 |

- **Pata de normalización: FNV (+1).**
- Spread de Johansen: ADF −3.53, p = 0.0072, **vida media de 58.5 días**.

**Filtro 1:**
- q = 5.62·10⁻⁶ (Q/R = 3.5·10⁻⁵, por el respaldo); R = 0.01298 (√R = 0.114).
- Diagonal de Q: NEM 7.1·10⁻⁷, KGC 2.2·10⁻⁷, GFI 2.9·10⁻⁷, HMY 1.3·10⁻⁷, RGLD 9.2·10⁻⁷.
- P₀ = 10⁴·I; burn-in de 842 días (hasta 2014-05-09).
- **Piso de P** (mediana después del burn-in): de 2.9·10⁻⁴ a 9.5·10⁻⁴.
- **Robustez a P₀** (10², 10⁴, 10⁶): los pesos difieren como máximo 4.2·10⁻⁵ después del burn-in.
- Spread del filtro (train, después del burn-in): ADF p = 0.034, **vida media de 55.0 días**, **7 cambios de signo** de pesos (NEM 1, KGC 1, GFI 3, HMY 2).

**Respaldos usados: 2.** Uno en la etapa 2 (ningún subgrupo pasa la correlación) y otro en q (ningún q cumple "sin cambios de signo").

## Dirección de las posiciones (leer con cuidado)

**Estar largo el spread significa comprar cada activo con β_i > 0 y vender cada activo con β_i < 0, en proporción a |β_i|. Estar corto el spread invierte todos los signos. El nocional se reparte entre todos los activos en proporción a |β_i|.**

- Con el β de train: **largo el spread = comprar FNV, GFI y KGC; vender RGLD, NEM y HMY.**
- **Pata de normalización:** FNV (β = +1). **Convención de signo:** la de arriba, con β̂_t de `beta_t.csv` (FNV siempre en +1).
- **Vida media del spread: 58.5 días (Johansen) y 55.0 días (Kalman), ambas en train.** Úsenla para calibrar el periodo máximo de tenencia.

> **Advertencia:** un error de signo produce una estrategia que pierde de forma sostenida y parece mala suerte. Arturo y Paula deben **verificar la dirección sobre un episodio conocido** (por ejemplo, un día de train en que el spread esté muy por arriba de su media: la posición debe ser corta en el spread y debe ganar cuando el spread baja).

## Archivos que produzco

| Archivo | Contenido |
|---|---|
| `data/universe.json` | Configuración del universo y su justificación |
| `data/raw/prices_raw.csv`, `download_meta.json` | Descarga congelada |
| `data/processed/prices.csv` | Los **9** activos del universo, precios ajustados en USD + `period`. **Ojo:** el grupo final son 6; los tickers están en `selected_group.json` |
| `data/processed/splits.json`, `cleaning_report.json` | Partición y conteos de validación |
| `artifacts/cointegration/group_selection.csv` | Los 84 subgrupos: etapa alcanzada, estadísticos, decisión y razón |
| `artifacts/cointegration/selected_group.json` | Tickers, r̂, eigenvalores, traza y máximo eigenvalor con valores críticos, todas las columnas de β, columna elegida, pata, signo, ADF, vida media, orden p y especificación |
| `artifacts/cointegration/stability_windows.csv` | Etapa 7 por sub-ventana |
| `artifacts/cointegration/eigvec_rolling.csv` | Primer eigenvector móvil (solo datos pasados) + `period` |
| `artifacts/cointegration/intento1_energia_etapa2.csv` | Registro del primer universo descartado |
| `artifacts/kalman/beta_t.csv` | β̂_t normalizado por activo (FNV = +1) + `period`, `burn_in` |
| `artifacts/kalman/spread_t.csv` | `spread` (S_t = β̂_tᵀ(P_t − μ) = ỹ_t), `innovation_normalized`, `innovation_std` (√S_t), `period`, `burn_in` |
| `artifacts/kalman/gain_1.csv`, `posvar_1.csv` | K_t y diagonal de P_{t\|t} por peso |
| `artifacts/kalman/filter1_q_grid.csv`, `filter1_params.json` | Rejilla de q (29) y parámetros finales |
| `artifacts/figures/fig1_log_precios.png`, `fig3_beta_t.png`, `fig4_eigenvector.png`, `fig_diagnostico_filtro1.png` | Figuras 1, 3, 4 y diagnóstico (train) |

Todas las salidas de `artifacts/kalman/` tienen la fecha ISO en la primera columna, columnas en minúsculas y la columna `period`.

## Supuestos

- Los precios ajustados de Yahoo son correctos; no se encontró ningún error que corregir.
- La constante del sistema es un término determinístico (la media de train), no un estado.
- Train se trata como "en muestra": q, R, μ y ẑ₀ usan todo train. Test y valid no se tocaron.

## Límites conocidos

1. **Evidencia de cointegración débil y compatible con el azar:** 1 de 36 pares pasa Engle-Granger (se esperan 1.8) y 2 de 30 subgrupos dan r̂ ≥ 1 (se esperan 1.5). El elegido es el nº 30 de la lista.
2. **La etapa 2 no se cumple** (0 de 84) y se usó un respaldo.
3. **Estabilidad baja:** cointegrado en el 14% de las sub-ventanas de train. Una ruptura en test o valid es probable.
4. **`test_beta_stability` falla:** 7 cambios de signo en train, porque KGC y HMY tienen pesos casi nulos. No se relajó el umbral.
5. **`pytest` a secas no encuentra los paquetes** sin un `pytest.ini` (`pythonpath = .`) en la raíz. Ese archivo es de la estructura del repo y no lo creé. `python -m pytest` sí funciona.
6. **Vida media cerca del límite de 60 días.** El burn-in ocupa el 37% de train.
7. **Data snooping:** dos universos probados en esta versión y 2 × 8,008 grupos de petroleras en versiones anteriores, sobre el mismo train.
