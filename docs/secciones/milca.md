# Datos, selección del grupo por cointegración y filtro de Kalman 1 (Milca)

Todo lo que se estima o elige en esta sección usa **solo la partición de entrenamiento (train, 2011-01-03 a 2019-12-30)**. Cada cifra indica su tramo. El análisis trabaja con el **logaritmo natural** de los precios ajustados.

> **Notación.** En el filtro de Kalman, z_t es el estado oculto (los pesos β_t) y S_t es la varianza de la innovación. En Johansen, S_t = βᵀP_t es el spread. El z-score de las señales (Arturo) es otra cosa.

## 1. Universo de activos

### 1.1 Sector y por qué

**Cadena de valor del oro:** minería (grandes, medianas y africanas) y financiamiento minero por regalías y *streaming*.

El precio del oro es una **tendencia estocástica común**, observable y dominante: es el principal determinante del ingreso de las nueve empresas.
- Las mineras venden un producto homogéneo (onzas de oro) a un precio global único.
- Las empresas de regalías y *streaming* cobran un porcentaje de esas mismas onzas.

Otros sectores (bancos, tecnología, consumo) dependen de varios factores distintos (tasas, crédito, innovación, demanda) y no tienen un solo motor común tan claro.

**Por qué solo 9:** mientras más activos se prueban, mayor es la probabilidad de falsos positivos. Nueve activos del mismo sector dejan C(9,6) = **84 subgrupos**, y cualquier relación encontrada tiene una razón económica detrás.

### 1.2 Los 9 activos (configuración en `data/universe.json`)

| Ticker | Empresa | Parte de la cadena | Justificación económica |
|---|---|---|---|
| NEM | Newmont | Minería: productora grande, global | La mayor productora de oro; su ingreso es casi todo venta de oro a precio spot. Competidora directa de AEM y KGC |
| AEM | Agnico Eagle | Minería: productora grande en jurisdicciones estables | Mismo producto y mismo precio global que NEM. Doble listado TSX/NYSE |
| KGC | Kinross Gold | Minería: mediana-grande (América, África occidental) | Mayor apalancamiento operativo: se mueve más que proporcionalmente con el oro, pero en la misma dirección. Doble listado TSX/NYSE |
| AU | AngloGold Ashanti | Minería: grande, con exposición africana | Comparte el precio del oro como motor y compite con NEM. ADR (doble listado JSE/NYSE) |
| GFI | Gold Fields | Minería: grande (Sudáfrica, Ghana, Australia, Perú) | Misma tendencia común que AU y HMY. ADR (JSE/NYSE) |
| HMY | Harmony Gold | Minería: sudafricana de costos altos | Alto apalancamiento al oro; competidora de AU y GFI. ADR (JSE/NYSE) |
| FNV | Franco-Nevada | Financiamiento: regalías y *streaming* | Cobra un porcentaje de la producción de terceros sin los costos de operar. Otra parte de la cadena con el mismo motor. Doble listado TSX/NYSE |
| WPM | Wheaton Precious Metals | Financiamiento: *streaming* de oro y plata | Compra metal a precio fijo bajo; su margen sigue al oro y a la plata. Competidora de FNV y RGLD |
| RGLD | Royal Gold | Financiamiento: regalías y *streaming* | Modelo igual al de FNV y WPM; ingreso ligado al oro |

**Por qué se espera un equilibrio de largo plazo:** el valor de cada empresa es, en esencia, onzas por precio del oro, menos costos. Si una se aleja mucho de las demás sin un cambio en sus onzas o costos, el mercado debería corregir la diferencia.

**Cambio forzado por datos:** Barrick Gold (GOLD) se excluyó porque Yahoo solo tiene su historia desde 2014-03-17 (cambió de ticker ABX a GOLD). Se reemplazó por Harmony Gold (HMY), que sí tiene historia completa. La decisión se tomó por disponibilidad de datos, no por rendimientos.

### 1.3 Transparencia: data snooping en la elección del universo

- **Primer intento (registrado en `artifacts/cointegration/intento1_energia_etapa2.csv`):** 9 productoras de petróleo (IMO, SU, CVE, CNQ, DVN, OVV, APA, MUR, OXY). **Ninguno de sus 84 subgrupos pasó** el filtro de correlación de retornos (máximo 0.647 < 0.70). Después de verlo se cambió al oro, que también falló el filtro (máximo 0.652). Elegir el universo después de un fracaso es un grado de libertad del investigador, y esos 84 subgrupos cuentan como pruebas.
- **Versiones anteriores del equipo:** se evaluaron 2 × 8,008 grupos de 16 petroleras sobre **este mismo periodo de train**. No se usaron aquí, pero forman parte del historial de pruebas.

## 2. Datos (`data/`)

- **Fuente:** Yahoo Finance (yfinance 1.7.0), cierre ajustado por dividendos y splits. Congelado en `data/raw/prices_raw.csv`.
- **Ventana:** 2011-01-01 a 2025-12-31 (15 años, el máximo permitido). Da la muestra de train más larga posible y termina en un año cerrado. Resultan 3,772 días hábiles, de 2011-01-03 a 2025-12-31.
- **Validación:** se aplica a toda la muestra, porque es limpieza y no estimación.

| Caso | Encontrados | Regla de resolución |
|---|---|---|
| Valores faltantes | 0 (en los 9 activos) | La fecha se descarta para todos (no se rellena: rellenar inventa precios) |
| Fechas duplicadas | 0 | Se conserva la primera aparición |
| Fechas desalineadas entre activos | 0 | La fecha se descarta completa |
| Precios no positivos | 0 | Se tratan como faltantes |
| Saltos anómalos (\|log-retorno\| > 0.40) | 0 | Se marcan sin modificar; "posible error" si revierten más de la mitad al día siguiente. Los precios ya vienen ajustados por splits |

- **Partición cronológica 60/20/20** (`data/processed/splits.json`):

| Tramo | Inicio | Fin | Días |
|---|---|---|---|
| train | 2011-01-03 | 2019-12-30 | 2,263 |
| test | 2019-12-31 | 2022-12-27 | 754 |
| valid | 2022-12-28 | 2025-12-31 | 755 |

## 3. Marco teórico

**Eigenvalores y eigenvectores.** Un eigenvector de una matriz es un vector que, al transformarse con esa matriz, **no cambia de dirección, solo de escala**. El factor de escala es su eigenvalor. Johansen calcula eigenvalores y eigenvectores de una matriz construida con los datos, de forma parecida a PCA: reduce la dimensionalidad buscando las combinaciones de activos más estables. **Los eigenvectores indican cuánto invertir en cada activo.**

**Spread del portafolio.** S_t = βᵀ·P_t, con P_t el vector de **log-precios** de los n activos en t. β es el vector cuya combinación tiene **reversión a la media**.

**Pruebas múltiples.** Mientras más activos se prueban, mayor es la probabilidad de falsos positivos. Por eso los 9 candidatos vienen del mismo sector.

**Del VAR(p) al VECM.** Se parte de

P_t = A_1·P_{t−1} + A_2·P_{t−2} + … + A_p·P_{t−p} + ε_t

que se reescribe como modelo de corrección de error vectorial:

ΔP_t = Π·P_{t−1} + Σ_{i=1}^{p−1} Γ_i·ΔP_{t−i} + ε_t

- A_i es la matriz de coeficientes: el aporte de cada rezago de cada activo.
- I es la matriz identidad.
- p es cuántos niveles de autorregresión hay.

- **p = 1:** restando P_{t−1} de ambos lados, ΔP_t = (A_1 − I)·P_{t−1} + ε_t.
- **p = 2:** ΔP_t = (A_1 − I)·P_{t−1} + A_2·P_{t−2} + ε_t. Como P_{t−2} = P_{t−1} − ΔP_{t−1}:
  ΔP_t = (A_1 + A_2 − I)·P_{t−1} − A_2·ΔP_{t−1} + ε_t.
- **p = 3:** ΔP_t = (A_1 − I)·P_{t−1} + A_2·P_{t−2} + A_3·P_{t−3} + ε_t. Con P_{t−2} = P_{t−1} − ΔP_{t−1} y P_{t−3} = P_{t−1} − ΔP_{t−1} − ΔP_{t−2}:
  ΔP_t = (A_1 + A_2 + A_3 − I)·P_{t−1} − (A_2 + A_3)·ΔP_{t−1} − A_3·ΔP_{t−2} + ε_t.
  Es decir, Π = A_1 + A_2 + A_3 − I, Γ_1 = −(A_2 + A_3) y Γ_2 = −A_3.

**Estimación.** Por **máxima verosimilitud**: se busca la combinación de parámetros que mejor explica los datos observados. La **prueba de razón de verosimilitud (LR)** determina r, el número de eigenvectores cointegrados. Hay dos versiones:
- **Traza:** ¿hay más de r relaciones?
- **Máximo eigenvalor:** ¿hay exactamente r + 1 relaciones contra r?

**Condición de rango.** El rango de Π es el número r de combinaciones estacionarias, se escribe Π = αβᵀ, y las r columnas de β son los vectores de cointegración.
- **r = 0 (alarma):** Π = 0, no hay ninguna combinación estacionaria. Los precios no están atados y el grupo no sirve para reversión a la media.
- **r = n (alarma):** Π tiene rango completo, así que cada serie sería estacionaria por sí sola. No hay tendencia común que compartir, y casi siempre indica un problema de datos o de especificación.
- **0 < r < n:** hay r relaciones de equilibrio y n − r tendencias comunes. Es el caso que se busca.

## 4. Embudo de selección (7 etapas, solo train, log-precios)

Resultados por subgrupo en `artifacts/cointegration/group_selection.csv` (84 filas: etapa alcanzada, estadísticos, decisión y razón).

**Etapa 1 (universo):** los 9 activos de la sección 1.

**Etapa 2 (correlación móvil sobre retornos; heurística, no criterio definitivo):**
- Log-retornos diarios, ventana de 63 días (un trimestre).
- Train dividido en 3 sub-ventanas de ~3 años, del mismo largo que test y valid.
- Criterio: correlación media por pares > 0.70 en las tres sub-ventanas.
- **Resultado: 0 de 84 pasan.** El máximo de la correlación mínima por sub-ventana es **0.652** (nem-aem-kgc-fnv-wpm-rgld).
- **Respaldo (contado: 1 vez):** como la correlación es una heurística y el criterio definitivo es Johansen, la etapa 3 prioriza los 84. **Esta regla se escribió después de ver que nadie pasaba.**

**Etapa 3 (Engle-Granger por pares; prioriza, no elimina):**
- Para cada subgrupo, sus 15 pares (OLS + ADF de residuales, p < 0.05). Son 36 pares distintos en total y **solo 1 pasa** (KGC-HMY, p = 0.0015).
- 35 subgrupos tienen 1 par que pasa (los que contienen KGC y HMY) y 49 tienen 0.
- Orden: más pares que pasan y, de desempate, menor p-value medio. Ningún subgrupo se eliminó.

**Etapas 4 a 6 (regla secuencial, escrita antes de ver resultados):** se toma el subgrupo número 1, se aplica Johansen y se validan sus columnas; si no cumple, se pasa al siguiente.
- **30 subgrupos llegaron a Johansen** (los puestos 1 a 30). Ese es el número que se usa en pruebas múltiples.
  - 28 dieron r̂ = 0 (alarma).
  - El nº 8 (kgc-gfi-hmy-fnv-wpm-rgld) dio r̂ = 1, pero no tenía columna operable.
  - **El nº 30, NEM-KGC-GFI-HMY-FNV-RGLD, cumple.**
- **Orden p del VAR:** AIC = 1 y BIC = 1 (train). Se usa BIC, regla fijada antes por parsimonia (cada rezago agrega 36 parámetros con 6 series). El VECM queda sin rezagos en diferencias.
- **Especificación determinística:** `det_order = 0`, **constante sin restringir (caso 3), fuera de la relación de cointegración**, en los términos determinísticos del VECM. Valores críticos de **MacKinnon, Haug y Michelis (1996)** (`statsmodels.tsa.coint_tables`). Es consistente con el filtro de Kalman 1 sin intercepto en el estado, que trata la constante igual: resta la media de train.

**Johansen del grupo elegido (train):**

| Hipótesis | Traza | VC 95% | Máx. eigenvalor | VC 95% |
|---|---|---|---|---|
| r = 0 | **99.75** | 95.75 | **45.96** | 40.08 |
| r ≤ 1 | 53.79 | 69.82 | 22.81 | 33.88 |
| r ≤ 2 | 30.98 | 47.85 | 15.44 | 27.59 |
| r ≤ 3 | 15.54 | 29.80 | 10.33 | 21.13 |
| r ≤ 4 | 5.21 | 15.49 | 4.96 | 14.26 |
| r ≤ 5 | 0.26 | 3.84 | 0.26 | 3.84 |

- **r̂ = 1, y la traza y el máximo eigenvalor coinciden.** No es ni 0 ni n, así que no hay alarma.
- La traza supera su valor crítico por poco: 99.75 contra 95.75.
- Eigenvalores: 0.02011, 0.01003, 0.00680, 0.00456, 0.00219, 0.00011.

**Etapa 5 (selección del vector, train):**

| Columna | Dentro de r̂ | ADF | p-value | Vida media (días) | Operable |
|---|---|---|---|---|---|
| **0** | **sí** | **−3.530** | **0.0072** | **58.5** | **sí (elegida)** |
| 1 | no | −2.642 | 0.0846 | 113.5 | no |
| 2 | no | −3.330 | 0.0136 | 84.1 | no |
| 3 | no | −2.909 | 0.0443 | 99.7 | no |
| 4 | no | −1.918 | 0.3237 | 204.9 | no |
| 5 | no | −2.887 | 0.0469 | 183.7 | no |

**Normalización:** el coeficiente de mayor |β| de la columna 0 es el de **FNV**, así que se divide entre él:

| | NEM | KGC | GFI | HMY | FNV | RGLD |
|---|---|---|---|---|---|---|
| β (train) | −0.314 | +0.041 | +0.474 | −0.057 | **+1** | −0.741 |

**Convención de signo:** largo el spread = comprar cada activo con β_i > 0 (FNV, GFI, KGC) y vender cada activo con β_i < 0 (RGLD, NEM, HMY), en proporción a |β_i|. Corto el spread invierte todos los signos.

**Etapa 6 (validación del spread, train):** ADF −3.53, p = **0.0072** (< 0.05) y vida media de **58.5 días**, dentro del rango de 5 a 60, pero **en su orilla superior**.

**Etapa 7 (estabilidad, train):**
- Las etapas 4 a 6 se repiten en 14 sub-ventanas de 504 días (2 años) que avanzan 126 días (medio año).
- Una sub-ventana "sigue cointegrada" si r̂ ≥ 1 y alguna columna dentro de r̂ es operable.

| Sub-ventana | r̂ | ADF p (mejor columna) | Vida media | ¿Cointegrado? |
|---|---|---|---|---|
| 2011-01 a 2013-01 | 0 | 0.0011 | 10.5 | no |
| 2011-07 a 2013-07 | 0 | 0.0000 | 7.2 | no |
| 2012-01 a 2014-01 | 0 | 0.0000 | 8.7 | no |
| 2012-07 a 2014-07 | 0 | 0.0000 | 8.5 | no |
| 2013-01 a 2015-01 | 1 | 0.0000 | 10.9 | **sí** |
| 2013-07 a 2015-07 | 0 | 0.0005 | 8.8 | no |
| 2014-01 a 2016-01 | 1 | 0.0003 | 8.4 | **sí** |
| 2014-07 a 2016-07 | 1 | 0.0691 | 19.5 | no |
| 2015-01 a 2017-01 | 0 | 0.0059 | 13.4 | no |
| 2015-07 a 2017-07 | 0 | 0.0563 | 19.8 | no |
| 2016-01 a 2018-01 | 0 | 0.0022 | 15.6 | no |
| 2016-07 a 2018-07 | 0 | 0.0000 | 7.2 | no |
| 2017-01 a 2019-01 | 0 | 0.0007 | 7.2 | no |
| 2017-07 a 2019-07 | 0 | 0.0001 | 7.7 | no |

**El grupo sigue cointegrado solo en 2 de 14 sub-ventanas (14%).** En las demás, Johansen no encuentra relación aunque la mejor combinación pase el ADF (ese ADF se calcula sobre un vector elegido con los mismos datos, así que está sesgado a favor).

**Pérdida de cointegración fuera de muestra.** Si la cointegración se pierde fuera de la muestra, la estrategia pierde sentido: el spread deja de regresar y las posiciones acumulan pérdidas. Aquí estamos **minimizando el error dentro de la muestra (train)**. Una ruptura posterior es posible, y con un 14% de sub-ventanas cointegradas en train es incluso probable. Se documenta en lugar de esconderse.

**Eigenvector móvil (figura 4):** `artifacts/cointegration/eigvec_rolling.csv` tiene el primer eigenvector de Johansen en cada fecha, con los 504 días anteriores (solo datos pasados), escalado a longitud 1 y con el signo alineado al β de train.

## 5. Filtro de Kalman 1 (`kalman/filter1_weights.py`)

### 5.1 Modelo (notación del profesor)

- **Estado:** z_t = β_t, los pesos de NEM, KGC, GFI, HMY y RGLD frente a FNV, la pata de normalización. **Sin intercepto α.**
- **Transición:** z_t = A·z_{t−1} + W_t, W_t ~ N(0, Q), con **A = I**. Q es la velocidad del drift de los pesos.
- **Observación:** y_t = C_t·z_t + V_t, V_t ~ N(0, R).
  - y_t = log P_FNV − μ_FNV; C_t = (log P_i − μ_i) de las otras 5.
  - μ = media de train de cada log-precio: el término determinístico, consistente con Johansen.
- **Predecir:** ẑ_{t|t−1} = A·ẑ_{t−1|t−1}; P_{t|t−1} = A·P_{t−1|t−1}·Aᵀ + Q.
- **Actualizar:** ỹ_t = y_t − C_t·ẑ_{t|t−1}; S_t = C_t·P_{t|t−1}·C_tᵀ + R; K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹; ẑ_{t|t} = ẑ_{t|t−1} + K_t·ỹ_t; P_{t|t} = (I − K_t·C_t)·P_{t|t−1}.
- **Pesos β̂_t normalizados:** +1 en FNV y −ẑ_{t|t−1} en las demás. Con los pesos previos (sin look-ahead), el spread del día es β̂_tᵀ(P_t − μ) = ỹ_t.

### 5.2 Selección de Q y R (train)

**R** = varianza residual de la regresión OLS estática de y_t sobre C_t en train (sin constante, log-precios centrados): **R = 0.01298**, o sea √R = 0.114. Equivale a un ruido de alrededor del **11.4% en el log-precio de FNV** que la canasta no explica en un día dado. Es grande porque R absorbe la desviación persistente del spread: el filtro trata esa desviación como ruido y no la persigue.

**Q** = q · Q_base, con Q_base = R · diag(1/E[C_i²]): cada componente aporta lo mismo a la predicción por unidad de q.
- **No se usó máxima verosimilitud libre:** en este modelo elige Q demasiado grande, los pesos persiguen cada desviación y el filtro se come la reversión.
- **Rejilla:** q de 10⁻⁸ a 10⁻¹, con **29 valores** log-espaciados (4 por década). Va de pesos casi congelados (Q/R ≈ 10⁻⁷) a pesos que persiguen cada desviación (Q/R ≈ 0.6).
- **Criterio, escrito antes de ver resultados** (train, después del burn-in de cada q):
  - (a) ADF p < 0.05;
  - (b) vida media entre 5 y 60 días;
  - (c) **ningún peso cambia de signo**;
  - (d) los pesos se adaptan: al final de train difieren al menos 0.05 de los que daría Q = 0;
  - si varios cumplen, el centro de la meseta.

**Rejilla de q (train, `artifacts/kalman/filter1_q_grid.csv`):**

| q | Q/R | Burn-in | ADF | ADF p | Vida media | Cambios de signo | Adaptación | var(ỹ/√S) | Autocorr. ỹ/√S | (a) | (b) | (c) | (d) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.0e-8 | 6.2e-8 | 398 | −3.24 | 0.018 | 58.1 | 10 | 0.001 | 1.16 | 0.987 | sí | sí | no | no |
| 1.0e-7 | 6.2e-7 | 413 | −3.22 | 0.019 | 58.5 | 10 | 0.011 | 1.16 | 0.988 | sí | sí | no | no |
| 3.2e-7 | 2.0e-6 | 483 | −3.11 | 0.026 | 60.5 | 7 | 0.029 | 1.17 | 0.988 | sí | no | no | no |
| 1.0e-6 | 6.2e-6 | 783 | −2.74 | 0.067 | 66.5 | 5 | 0.062 | 1.23 | 0.989 | no | no | no | sí |
| 1.8e-6 | 1.1e-5 | 939 | −2.80 | 0.058 | 60.7 | 4 | 0.083 | 1.19 | 0.989 | no | no | no | sí |
| 3.2e-6 | 2.0e-5 | 974 | −2.82 | 0.055 | 57.6 | 4 | 0.103 | 1.06 | 0.988 | no | sí | no | sí |
| **5.6e-6** | **3.5e-5** | **842** | **−3.01** | **0.034** | **55.0** | **7** | **0.121** | **0.96** | **0.988** | **sí** | **sí** | **no** | **sí** |
| 1.0e-5 | 6.2e-5 | 479 | −2.86 | 0.050 | 46.1 | 15 | 0.136 | 0.81 | 0.985 | sí | sí | no | sí |
| 3.2e-5 | 2.0e-4 | 400 | −3.29 | 0.015 | 34.8 | 9 | 0.154 | 0.62 | 0.980 | sí | sí | no | sí |
| 1.0e-4 | 6.2e-4 | 273 | −3.99 | 0.001 | 24.3 | 22 | 0.166 | 0.45 | 0.972 | sí | sí | no | sí |
| 1.0e-3 | 6.2e-3 | 509 | −5.73 | 7e-7 | 9.3 | 27 | 0.239 | 0.19 | 0.929 | sí | sí | no | sí |
| 1.0e-2 | 6.2e-2 | 410 | −10.8 | 2e-19 | 3.2 | 64 | 0.292 | 0.06 | 0.811 | sí | no | no | sí |
| 1.0e-1 | 0.62 | 261 | −24.2 | ≈ 0 | 1.2 | 113 | 0.314 | 0.02 | 0.573 | sí | no | no | sí |

(La tabla completa, con los 29 valores, está en el artefacto.)

- **Ningún q cumple (c).** Incluso con los pesos casi congelados hay entre 4 y 10 cambios de signo después del burn-in. Los cruces son suaves (|w| < 0.008 al cruzar) y vienen de patas con peso casi nulo en Johansen (KGC +0.04, HMY −0.06) y de pesos que siguen ajustándose. No es un problema de calibración de Q: es la composición del grupo.
- **Respaldo (contado: 1 vez; escrito después de ver que ningún q cumple (c)):** entre los q que cumplen (a), (b) y (d), el de menos cambios de signo. **q = 5.62·10⁻⁶** (Q/R = 3.5·10⁻⁵), con 7 cambios.
- **El q elegido no está en la orilla de la rejilla de 29 valores**, pero **sí en la orilla de la región que cumple (a), (b) y (d)**: el q anterior (3.2·10⁻⁶) reprueba el ADF.
- **Diagonal de Q final:** NEM 7.1·10⁻⁷; KGC 2.2·10⁻⁷; GFI 2.9·10⁻⁷; HMY 1.3·10⁻⁷; RGLD 9.2·10⁻⁷.

### 5.3 Inicialización, burn-in y convergencia (train)

- **P₀ = 10⁴·I.** Con el vector normalizado, todos los |β_i| son ≤ 1, así que una desviación de 100 es difusa. **ẑ₀** = β de Johansen de train.
- **Burn-in:** primer día desde el cual la varianza posterior de **todos** los componentes se queda por debajo del techo de su banda estable hasta el final de train. Banda = exp(mediana ± 3·MAD) del log de P_{t|t} en train. **Salió de 842 días (hasta 2014-05-09), el 37% de train.**
- **Convergencia:** P_t baja de 10⁴ y se estabiliza.
  - **Piso** (mediana después del burn-in): NEM 9.5·10⁻⁴; KGC 4.7·10⁻⁴; GFI 5.2·10⁻⁴; HMY 2.9·10⁻⁴; RGLD 7.7·10⁻⁴. Mínimo en train: 2.3·10⁻⁴.
  - **Con Q > 0, P converge a un piso mayor que cero y no a cero exacto**, porque el modelo supone que los pesos verdaderos sí se mueven: cada día el dato reduce la incertidumbre y Q la repone.
  - Solo con Q = 0 llegaría a cero, y entonces K → 0 y el filtro dejaría de aprender.
- **Robustez a P₀** (10², 10⁴ y 10⁶·I): el burn-in es el mismo (842 días). Después del burn-in, los pesos difieren como máximo 4.2·10⁻⁵ y el spread 7.1·10⁻⁶. **El resultado no depende de P₀.**
- **Diagnóstico de innovaciones** (q elegido, train, después del burn-in): var(ỹ/√S) = 0.96 y autocorrelación de orden 1 de 0.988. En el filtro 1 el spread **es** la innovación, así que esa autocorrelación es la reversión lenta (vida media de 55 días) que se opera.

### 5.4 Ganancia de Kalman: cálculo e interpretación

K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹ dice **cuánto se mueve cada peso por cada unidad de sorpresa ỹ_t**. Es grande cuando la incertidumbre de los pesos (P) es grande frente a la incertidumbre total de la predicción (S).
- Después del burn-in, |K| es del orden de 0.003 a 0.03 (figura de diagnóstico, escala lineal).
- Su signo cambia según la dirección de C_t. Por eso se grafica en valor absoluto y escala lineal: la escala log escondería los negativos.

**Ejemplo:** el 2014-05-09, ỹ = −0.0653 y K_NEM = −0.0282, así que ẑ_NEM se mueve (−0.0282)(−0.0653) = +0.0018, de −0.0818 a −0.0800. Con un q tan chico, el filtro ajusta los pesos muy despacio.

### 5.5 Ejemplo numérico de 5 periodos (train, los primeros días después del burn-in)

Recalculado a mano con la matriz P completa y las fórmulas de 5.1. **La diferencia máxima contra el código es 0.** Datos: μ = (NEM 3.27730, KGC 1.46261, GFI 1.38020, HMY 1.05497, FNV 3.90013, RGLD 4.10620), R = 0.012985 y Q con la diagonal de 5.2.

y_t y C_t (log-precios centrados):

| Fecha | y_t (FNV) | NEM | KGC | GFI | HMY | RGLD |
|---|---|---|---|---|---|---|
| 2014-05-09 | −0.146610 | −0.355635 | −0.170249 | −0.292595 | −0.104416 | −0.087417 |
| 2014-05-12 | −0.147018 | −0.335441 | −0.162869 | −0.287570 | −0.074954 | −0.071036 |
| 2014-05-13 | −0.166571 | −0.328529 | −0.170249 | −0.305270 | −0.091214 | −0.089734 |
| 2014-05-14 | −0.146406 | −0.330557 | −0.177684 | −0.302722 | −0.058953 | −0.097656 |
| 2014-05-15 | −0.150897 | −0.344457 | −0.190200 | −0.302722 | −0.081426 | −0.096097 |

ẑ_{t|t−1} (predicción):

| Fecha | NEM | KGC | GFI | HMY | RGLD |
|---|---|---|---|---|---|
| 2014-05-09 | −0.081798 | 0.013749 | 0.251714 | −0.367515 | 0.832490 |
| 2014-05-12 | −0.079959 | 0.014043 | 0.251513 | −0.367962 | 0.831773 |
| 2014-05-13 | −0.078410 | 0.014427 | 0.251725 | −0.368698 | 0.831058 |
| 2014-05-14 | −0.077166 | 0.014876 | 0.252264 | −0.369597 | 0.830481 |
| 2014-05-15 | −0.076681 | 0.015169 | 0.252637 | −0.370143 | 0.830243 |

Diagonal de P_{t|t−1}, innovación y su varianza:

| Fecha | P_{t\|t−1} (NEM, KGC, GFI, HMY, RGLD) | ỹ_t | S_t |
|---|---|---|---|
| 2014-05-09 | 0.003261, 0.001168, 0.003422, 0.001593, 0.001010 | −0.065310 | 0.013092 |
| 2014-05-12 | 0.003252, 0.001168, 0.003422, 0.001592, 0.001009 | −0.067718 | 0.013088 |
| 2014-05-13 | 0.003245, 0.001168, 0.003422, 0.001591, 0.001009 | −0.072087 | 0.013078 |
| 2014-05-14 | 0.003242, 0.001168, 0.003422, 0.001589, 0.001009 | −0.033592 | 0.013090 |
| 2014-05-15 | 0.003240, 0.001167, 0.003420, 0.001585, 0.001009 | −0.048302 | 0.013090 |

Ganancia K_t:

| Fecha | NEM | KGC | GFI | HMY | RGLD |
|---|---|---|---|---|---|
| 2014-05-09 | −0.028168 | −0.004504 | 0.003069 | 0.006841 | 0.010984 |
| 2014-05-12 | −0.022864 | −0.005667 | −0.003126 | 0.010874 | 0.010556 |
| 2014-05-13 | −0.017255 | −0.006225 | −0.007472 | 0.012471 | 0.008013 |
| 2014-05-14 | −0.014437 | −0.008728 | −0.011109 | 0.016254 | 0.007074 |
| 2014-05-15 | −0.019511 | −0.008057 | −0.005332 | 0.012856 | 0.008187 |

ẑ_{t|t} (corrección) y diagonal de P_{t|t}:

| Fecha | NEM | KGC | GFI | HMY | RGLD | P_{t\|t} (diagonal) |
|---|---|---|---|---|---|---|
| 2014-05-09 | −0.079959 | 0.014043 | 0.251513 | −0.367962 | 0.831773 | 0.003251, 0.001168, 0.003422, 0.001592, 0.001008 |
| 2014-05-12 | −0.078410 | 0.014427 | 0.251725 | −0.368698 | 0.831058 | 0.003245, 0.001168, 0.003422, 0.001591, 0.001008 |
| 2014-05-13 | −0.077166 | 0.014876 | 0.252264 | −0.369597 | 0.830481 | 0.003242, 0.001168, 0.003421, 0.001589, 0.001008 |
| 2014-05-14 | −0.076681 | 0.015169 | 0.252637 | −0.370143 | 0.830243 | 0.003240, 0.001167, 0.003420, 0.001585, 0.001008 |
| 2014-05-15 | −0.075739 | 0.015558 | 0.252894 | −0.370764 | 0.829848 | 0.003235, 0.001166, 0.003420, 0.001583, 0.001008 |

**Cómo se lee:**
- Como A = I, el ẑ_{t|t} de un día es el ẑ_{t+1|t} del siguiente.
- ỹ_t es negativo los 5 días (FNV está "barata" frente a la canasta): la desviación persiste, que es la reversión lenta que se opera.
- P baja muy poco cada día, porque el filtro ya está cerca de su piso.

### 5.6 Formulación de Powell (filtro 1)

**Proceso de modelado de 6 pasos:**

1. **Definir el problema:** estimar cada día los pesos que hacen estacionario al spread de los 6 activos, sin usar información futura y sin destruir la reversión.
2. **Modelar el estado:** z_t = β_t (5 pesos libres), con la creencia (ẑ_{t|t}, P_{t|t}).
3. **Modelar la transición:** caminata aleatoria z_t = I·z_{t−1} + W_t, con W_t ~ N(0, Q) y Q = q·Q_base.
4. **Modelar la observación:** y_t = C_t·z_t + V_t, con V_t ~ N(0, R), sobre log-precios centrados.
5. **Definir la regla de decisión:** publicar β̂_t = (+1, −ẑ_{t|t−1}) para el día t (pesos previos); el spread es ỹ_t.
6. **Definir el mecanismo de aprendizaje:** las ecuaciones de actualización de Kalman (ganancia K_t), con q elegido por la rejilla y R por OLS en train.

**Cinco elementos del modelo matemático:**

| Elemento | Filtro 1 |
|---|---|
| Variables de estado | Creencia (ẑ_{t\|t}, P_{t\|t}); información: log-precios hasta t y μ de train; física: ninguna |
| Variables de decisión | β̂_{t+1} = (+1, −A·ẑ_{t\|t}), los pesos publicados para el día siguiente |
| Información exógena | Log-precios de t+1, de donde salen y_{t+1} y C_{t+1} |
| Función de transición | Predicción y actualización de Kalman (sección 5.1) |
| Función objetivo | Minimizar el error esperado de estimación de los pesos, E[Σ_t ‖z_t − ẑ_{t\|t−1}‖²], sujeto a que el spread sea operable (estacionario, vida media de 5 a 60 días, signos estables y pesos que se adaptan) |

## 6. Figuras (`artifacts/figures/`)

- **Figura 1** (`fig1_log_precios.png`): log-precios de los 6 activos.
- **Figura 3** (`fig3_beta_t.png`): β̂_t por activo, con el β de Johansen de train como referencia.
- **Figura 4** (`fig4_eigenvector.png`): primer eigenvector móvil de Johansen.
- **Diagnóstico** (`fig_diagnostico_filtro1.png`, **solo train**): P_t en escala log y |K_t| en escala **lineal**.

Las figuras 1, 3 y 4 cubren 2011-2025 con las fronteras train/test/valid dibujadas y etiquetadas. Se revisaron visualmente con vistas recortadas a train, y no se calcula ni se reporta ninguna cifra de test ni de valid.

## 7. Limitaciones

1. **El filtro de correlación (etapa 2) no lo pasa ningún subgrupo**, en dos universos del mismo sector. Se usó un respaldo, escrito después de ver el fracaso.
2. **La evidencia de cointegración es débil y compatible con el azar:**
   - Engle-Granger: 1 de 36 pares distintos pasa; por azar al 5% se esperan **1.8**.
   - Johansen: 2 de 30 subgrupos dan r̂ ≥ 1; por azar se esperan **1.5**.
   - El grupo elegido es el número 30 de la lista. Con una corrección de Bonferroni para 30 pruebas (p < 0.0017), no sobreviviría: su traza apenas supera el valor crítico al 95%.
3. **Data snooping:**
   - En esta versión: 2 universos (2 × 84 subgrupos), 30 pruebas de Johansen y 29 configuraciones de q.
   - En versiones anteriores, sobre el mismo train: 2 × 8,008 grupos de petroleras.
   - La corrección formal para el desempeño resultante es el **Deflated Sharpe Ratio** (Bailey y López de Prado, 2014) o el **Reality Check** (White, 2000).
4. **Estabilidad baja:** cointegrado en 2 de 14 sub-ventanas de train (14%). Una ruptura fuera de muestra es probable.
5. **Vida media en la orilla:** 58.5 días con Johansen y 55.0 con Kalman, cerca del máximo de 60.
6. **El criterio (c) de signos del filtro no se cumple** con ningún q, y la prueba obligatoria `test_beta_stability` falla (7 cambios de signo en train). Se usó un respaldo para q.
7. **Burn-in largo** (842 días, el 37% de train).
8. **Train es en muestra** para el filtro (q, R, μ y ẑ₀ usan todo train).
9. **Sesgo de supervivencia** limitado: se excluyó Barrick por datos.
