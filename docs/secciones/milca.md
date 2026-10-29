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

### 1.3 Transparencia: data snooping en la elección del universo (desviación declarada)

- **Se probaron dos sectores.** Primero 9 productoras de petróleo (IMO, SU, CVE, CNQ, DVN, OVV, APA, MUR, OXY; registro en `artifacts/cointegration/intento1_energia_etapa2.csv`): ninguno de sus 84 subgrupos pasó la correlación de retornos (máximo **0.647**). **El oro se eligió después de que el primero fallara.** Es un grado de libertad del investigador y se cuenta en las pruebas múltiples (sección 7).
- **Versiones anteriores del equipo:** 2 × 8,008 grupos de 16 petroleras sobre **este mismo periodo de train**.

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
- Log-retornos diarios, ventana de 63 días, 3 sub-ventanas de ~3 años.
- Criterio: correlación media por pares > 0.70 en las tres.
- **Ningún subgrupo llega a 0.70 con retornos diarios, en ninguno de los dos universos.** Máximo de la correlación mínima por sub-ventana: **0.647 en energía** y **0.652 en oro** (nem-aem-kgc-fnv-wpm-rgld). El umbral no se cambió.
- **Regla de respaldo (contada: 1 vez):** como la correlación es una heurística y el criterio definitivo es Johansen, la etapa 3 prioriza los 84. **Esta regla se adoptó DESPUÉS de ver el fracaso y se declara como desviación.**
- **Dato solo informativo, no sustituye el criterio** (el profesor pidió retornos): la correlación media por pares de los **log-precios en niveles** en train llega a **0.865** como máximo (media de los 84: 0.467). Sale mucho más alta porque dos series con tendencia se correlacionan aunque no tengan relación (columna `info_corr_log_levels`).

**Etapa 3 (Engle-Granger por pares; prioriza, no elimina):** de 36 pares distintos solo pasa 1 (KGC-HMY, p = 0.0015). 35 subgrupos tienen 1 par que pasa y 49 tienen 0. Ningún subgrupo se eliminó.

**Etapas 4 a 6 (regla secuencial, escrita antes):**
- **30 subgrupos llegaron a Johansen:** 28 dieron r̂ = 0 (alarma), el nº 8 no tuvo columna operable, y **el nº 30 (NEM-KGC-GFI-HMY-FNV-RGLD) cumplió**.
- Orden p: AIC = 1 y BIC = 1; se usa BIC.
- Especificación: **constante sin restringir (caso 3), fuera de la relación de cointegración**. Valores críticos de **MacKinnon, Haug y Michelis (1996)**. Es consistente con el filtro 1 sin intercepto.

**Etapa 7 (estabilidad):**
- 14 sub-ventanas de 504 días que avanzan 126.
- Una sub-ventana es **estable** si el grupo sigue cointegrado (r̂ ≥ 1 por la traza) **y** el spread de la mejor columna dentro de r̂ pasa ADF (p < 0.05).
- **La estabilidad entra como criterio de selección** (sección 4.8), no solo como dato.

### 4.8 Selección final del grupo (solo train)

**Diagnóstico de patas del grupo del embudo:**

| Pata | \|β\| normalizado | Participación en el nocional |
|---|---|---|
| FNV | 1.000 | 38.1% |
| RGLD | 0.741 | 28.2% |
| GFI | 0.474 | 18.1% |
| NEM | 0.314 | 12.0% |
| HMY | 0.057 | **2.2%** |
| KGC | 0.041 | **1.6%** |

**Umbral de peso económicamente relevante, fijado antes de ver más resultados: 5% del nocional** (|β_i| / Σ|β|).
- Con 6 activos el peso igual sería 16.7%; menos de un tercio de eso no aporta exposición.
- Con el 80% de $1,000,000, el 5% son $40,000. Con ese tamaño, la comisión de 0.125% por lado anula cualquier aporte de la pata.
- **Quedan por debajo: HMY (2.2%) y KGC (1.6%).**
- Un peso cercano a cero oscila alrededor del cero y cambia de signo por puro ruido. Eso explica los 7 cambios de signo que hacían fallar `test_beta_stability`.
- El enunciado permite grupos de 2 a 8 activos, y el profesor anticipó que "puede que encontremos 6 pero solo 4 o 2 pasen". **Reducir el grupo a las patas relevantes no cambia ningún umbral.**

**Candidatos (4 en total):**
- (a) el grupo del embudo;
- (b) quitarle 1, 2 y 3 de sus patas de menor |β| (mínimo 3 activos);
- (c) los subgrupos que llegaron a Johansen y cumplieron las etapas 5 y 6: solo fue el mismo grupo del embudo, así que no agrega ninguno nuevo.

**Regla de selección, escrita antes de mirar la tabla:**
1. **Filtro obligatorio:**
   - r̂ ≥ 1 con la traza **y** con el máximo eigenvalor al 95%;
   - ADF del spread con p < 0.05;
   - vida media entre 5 y 60 días;
   - existe un q que cumple los criterios (a)-(d) del filtro 1, con **cero** cambios de signo.
2. Entre los elegibles: mayor fracción de sub-ventanas estables.
3. Desempate: ADF más negativo.
4. Segundo desempate: más activos.
5. Si ninguno es elegible, no se relaja nada.

**Tabla de candidatos (train, `artifacts/cointegration/candidatos_final.csv`):**

| | **6: NEM-KGC-GFI-HMY-FNV-RGLD** | **5: NEM-GFI-HMY-FNV-RGLD** | **4: NEM-GFI-FNV-RGLD** | **3: GFI-FNV-RGLD** |
|---|---|---|---|---|
| Origen | Grupo del embudo | Quitar KGC | Quitar KGC y HMY | Quitar KGC, HMY y NEM |
| p (AIC / BIC) | 1 / 1 | 1 / 1 | 1 / 1 | 1 / 1 |
| r̂ traza / máx. eig. (¿coinciden?) | 1 / 1 (sí) | 1 / 1 (sí) | 1 / 1 (sí) | **0** / 1 (**no**) |
| Traza r=0 (VC 95%) | 99.75 (95.75) | 84.44 (69.82) | 49.92 (47.85) | 29.43 (29.80) |
| Máx. eig. r=0 (VC 95%) | 45.96 (40.08) | 43.89 (33.88) | 34.75 (27.59) | 22.00 (21.13) |
| Eigenvalores | 0.0201, 0.0100, 0.0068, 0.0046, 0.0022, 0.0001 | 0.0192, 0.0098, 0.0058, 0.0022, 0.0000 | 0.0152, 0.0036, 0.0029, 0.0002 | 0.0097, 0.0032, 0.0001 |
| ADF por columna | −3.53, −2.64, −3.33, −2.91, −1.92, −2.89 | −3.53, −2.68, −2.13, −2.01, −2.52 | −3.47, −2.35, −2.22, −2.32 | −2.98, −2.44, −2.25 |
| Vida media por columna (días) | 58.5, 113.5, 84.1, 99.7, 204.9, 183.7 | 58.2, 112.6, 144.4, 190.9, 247.6 | 57.9, 180.2, 167.0, 330.5 | 77.0, 173.9, 370.5 |
| Columna elegida | 0 | 0 | 0 | ninguna |
| ADF del spread (p) | −3.530 (0.0072) | −3.529 (0.0073) | −3.467 (0.0089) | – |
| Vida media | 58.5 | 58.2 | 57.9 | – |
| Estabilidad (sub-ventanas) | 2/14 (14%) | 0/14 (0%) | **3/14 (21%)** | 1/14 (7%) |
| Pata de normalización | FNV | FNV | FNV | – |
| \|β\| | FNV 1, RGLD 0.741, GFI 0.475, NEM 0.314, HMY 0.057, KGC 0.041 | FNV 1, RGLD 0.736, GFI 0.491, NEM 0.308, HMY 0.031 | FNV 1, RGLD 0.577, GFI 0.512, NEM 0.411 | – |
| q que cumplen (a)-(d) / 29 | 0 (mín. 4 cambios de signo) | 0 (el único q sin cambios reprueba el ADF) | **2** | – |
| Cambios de signo con el q elegido | – | – | **0** (NEM 0, GFI 0, RGLD 0) | – |
| Burn-in / piso de P | – | – | 908 días / 1.8·10⁻⁴ a 4.6·10⁻⁴ | – |
| **Elegible** | no: sin q con cero cambios | no: sin q con cero cambios | **sí** | no: r̂ traza = 0 |

**Ganador: NEM, GFI, FNV, RGLD (4 activos)**, el **único elegible**. Además es el más estable de los cuatro (21%).

### 4.9 Grupo ganador: Johansen, vector y spread (train)

- p = 1 (AIC y BIC); constante sin restringir; valores críticos de MacKinnon, Haug y Michelis.

| Hipótesis | Traza | VC 95% | VC 99% | Máx. eigenvalor | VC 95% | VC 99% |
|---|---|---|---|---|---|---|
| r = 0 | **49.92** | 47.85 | 54.68 | **34.75** | 27.59 | 32.72 |
| r ≤ 1 | 15.17 | 29.80 | – | 8.09 | 21.13 | – |
| r ≤ 2 | 7.08 | 15.49 | – | 6.63 | 14.26 | – |
| r ≤ 3 | 0.45 | 3.84 | – | 0.45 | 3.84 | – |

- **r̂ = 1, con las dos pruebas de acuerdo.** No hay alarma (ni 0 ni n).
- La traza pasa al 95% pero **no al 99%**; el máximo eigenvalor sí pasa al 99%.
- Eigenvalores: 0.01525, 0.00357, 0.00293, 0.00020.
- **Etapa 5:** columna 0 (ADF −3.467, p = 0.0089, vida media 57.9). Las demás no son operables (p ≥ 0.15).
- **Normalización:** la pata de mayor |β| es **FNV** (+1).

| | NEM | GFI | FNV | RGLD |
|---|---|---|---|---|
| β (train) | −0.411 | +0.512 | **+1** | −0.577 |
| Participación en el nocional | 16.4% | 20.5% | 40.0% | 23.1% |

Todas las patas superan el 5%.

- **Convención de signo:** largo el spread = comprar FNV y GFI (β > 0) y vender RGLD y NEM (β < 0), en proporción a |β|. Corto el spread invierte todos los signos.
- **Etapa 6:** ADF p = **0.0089** < 0.05; vida media **57.9 días**, dentro de 5-60 pero cerca del límite.
- **Etapa 7:** estable en **3 de 14 sub-ventanas (21%)**.

| Sub-ventana | r̂ | ADF p | Vida media | Estable |
|---|---|---|---|---|
| 2011-01 a 2013-01 | 1 | 0.0018 | 11.2 | sí |
| 2011-07 a 2013-07 | 0 | 0.0143 | 14.9 | no |
| 2012-01 a 2014-01 | 0 | 0.0484 | 18.3 | no |
| 2012-07 a 2014-07 | 0 | 0.0148 | 16.0 | no |
| 2013-01 a 2015-01 | 2 | 0.0237 | 18.2 | sí |
| 2013-07 a 2015-07 | 1 | 0.0099 | 13.1 | sí |
| 2014-01 a 2016-01 | 0 | 0.0564 | 15.7 | no |
| 2014-07 a 2016-07 | 1 | 0.6492 | 41.7 | no |
| 2015-01 a 2017-01 | 0 | 0.1222 | 28.8 | no |
| 2015-07 a 2017-07 | 0 | 0.0948 | 22.4 | no |
| 2016-01 a 2018-01 | 0 | 0.0002 | 29.2 | no |
| 2016-07 a 2018-07 | 0 | 0.0037 | 10.1 | no |
| 2017-01 a 2019-01 | 0 | 0.0005 | 8.8 | no |
| 2017-07 a 2019-07 | 0 | 0.0003 | 9.3 | no |

**Pérdida de cointegración fuera de muestra.** Si la cointegración se pierde fuera de la muestra, la estrategia pierde sentido. Aquí estamos **minimizando el error dentro de la muestra (train)**, y una ruptura posterior es posible. Con un 21% de sub-ventanas estables, y ninguna después de 2015, es probable. Se documenta en vez de esconderse.

## 5. Filtro de Kalman 1 (`kalman/filter1_weights.py`)

### 5.1 Modelo (notación del profesor)

- **Estado:** z_t = β_t, los pesos de NEM, GFI y RGLD frente a FNV. **Sin intercepto α.**
- **Transición:** z_t = A·z_{t−1} + W_t, W_t ~ N(0, Q), con **A = I**. Q es la velocidad del drift.
- **Observación:** y_t = C_t·z_t + V_t, V_t ~ N(0, R).
  - y_t = log P_FNV − μ_FNV; C_t = (log P_i − μ_i) de NEM, GFI y RGLD.
  - μ = media de train de cada log-precio: el término determinístico, consistente con Johansen.
- **Predecir:** ẑ_{t|t−1} = A·ẑ_{t−1|t−1}; P_{t|t−1} = A·P_{t−1|t−1}·Aᵀ + Q.
- **Actualizar:** ỹ_t = y_t − C_t·ẑ_{t|t−1}; S_t = C_t·P_{t|t−1}·C_tᵀ + R; K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹; ẑ_{t|t} = ẑ_{t|t−1} + K_t·ỹ_t; P_{t|t} = (I − K_t·C_t)·P_{t|t−1}.
- **Pesos β̂_t:** +1 en FNV y −ẑ_{t|t−1} en las demás.
- **Spread** (sin look-ahead): β̂_tᵀ(P_t − μ) = ỹ_t.

**Aclaración sobre la regla de no mirar test ni valid:**
- El filtro se **corre** hacia adelante sobre toda la muestra con los parámetros **congelados en train**, porque Arturo y Paula necesitan esas series. Lo mismo vale para el eigenvector móvil (solo datos pasados en cada fecha) y para las figuras con fronteras de tramos.
- **Nada se estima, calibra ni elige mirando test o valid**, y ninguna cifra de esos tramos se usa para decidir en este bloque.
- Eso **no es look-ahead**.

### 5.2 Selección de Q y R (train)

- **R** = varianza residual de la regresión OLS estática de y sobre C (sin constante, log-precios centrados): **R = 0.01456**, o sea √R = 0.121. Equivale a un ruido de alrededor del **12% en el log-precio de FNV** no explicado por la canasta. R absorbe la desviación persistente del spread y por eso el filtro no la persigue.
- **Q** = q · Q_base, con Q_base = R · diag(1/E[C_i²]).
- **No se usó máxima verosimilitud libre:** elige Q demasiado grande y se come la reversión.
- **Rejilla:** 29 valores de q, de 10⁻⁸ a 10⁻¹ (4 por década).
- **Criterio, escrito antes:** (a) ADF p < 0.05; (b) vida media de 5 a 60 días; (c) **cero** cambios de signo; (d) pesos que se adaptan (al final de train difieren al menos 0.05 de los de Q = 0); centro de la meseta.

| q | Q/R | Burn-in | ADF | ADF p | Vida media | Cambios de signo | Adaptación | var(ỹ/√S) | Autocorr. | Cumple |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0e-8 | 8.8e-8 | 399 | −3.12 | 0.025 | 61.8 | 1 | 0.001 | 1.17 | 0.988 | no (b, c, d) |
| 1.0e-7 | 8.8e-7 | 417 | −3.10 | 0.027 | 62.0 | 1 | 0.008 | 1.17 | 0.988 | no (b, c, d) |
| 5.6e-7 | 4.9e-6 | 791 | −2.90 | 0.046 | 62.0 | 1 | 0.038 | 1.23 | 0.989 | no (b, c, d) |
| **1.0e-6** | **8.8e-6** | **908** | **−2.88** | **0.048** | **55.5** | **0** | **0.060** | **1.07** | **0.987** | **sí (elegido)** |
| 1.8e-6 | 1.6e-5 | 977 | −2.77 | 0.063 | 56.0 | 0 | 0.090 | 1.01 | 0.987 | no (a) |
| 3.2e-6 | 2.8e-5 | 841 | −3.08 | 0.028 | 52.4 | 0 | 0.127 | 0.93 | 0.987 | sí |
| 5.6e-6 | 4.9e-5 | 413 | −3.54 | 0.007 | 50.6 | 1 | 0.168 | 0.87 | 0.986 | no (c) |
| 1.0e-4 | 8.8e-4 | 320 | −3.83 | 0.003 | 28.3 | 2 | 0.272 | 0.45 | 0.976 | no (c) |
| 1.0e-3 | 8.8e-3 | 195 | −6.19 | 6e-8 | 10.4 | 10 | 0.262 | 0.20 | 0.936 | no (c) |
| 1.0e-2 | 8.8e-2 | 49 | −11.4 | 1e-20 | 3.6 | 40 | 0.331 | 0.07 | 0.829 | no (b, c) |
| 1.0e-1 | 0.88 | 44 | −13.8 | 8e-26 | 1.3 | 54 | 0.385 | 0.02 | 0.614 | no (b, c) |

(La tabla completa, con los 29 valores, está en `artifacts/kalman/filter1_q_grid.csv`.)

- **Cumplen 2 de 29:** q = 1.0·10⁻⁶ y q = 3.2·10⁻⁶.
- **No son contiguos** (el q intermedio, 1.8·10⁻⁶, reprueba el ADF con p = 0.063), así que **no hay una meseta**: hay dos rachas de largo 1, y la regla toma la primera. **q = 1.0·10⁻⁶** (Q/R = 8.8·10⁻⁶).
- No queda en la orilla de la rejilla.
- **Cero cambios de signo** (NEM 0, GFI 0, RGLD 0).
- **El ADF del spread con ese q es p = 0.0476: pasa, pero muy cerca del umbral.**
- **Diagonal de Q:** NEM 1.42·10⁻⁷; GFI 5.71·10⁻⁸; RGLD 1.83·10⁻⁷.
- **No hubo respaldos en q** (contador: 0).

### 5.3 Inicialización, burn-in y convergencia (train)

- **P₀ = 10⁴·I** (difuso: los |β| normalizados son ≤ 1). **ẑ₀** = β de Johansen de train.
- **Burn-in medido = 908 días (hasta 2014-08-13): el 40.1% de train.** Es consecuencia de un q pequeño: con Q chico, P tarda mucho en llegar a su banda y quedarse ahí.
- **Convergencia:** P_t baja de 10⁴ y se estabiliza.
  - **Piso** (mediana después del burn-in): NEM 4.6·10⁻⁴; GFI 1.8·10⁻⁴; RGLD 3.5·10⁻⁴. Mínimo en train: 1.3·10⁻⁴.
  - Con Q > 0, P converge a un piso **mayor que cero**, porque el modelo supone que los pesos verdaderos sí se mueven.
  - Solo con Q = 0 llegaría a cero, y el filtro dejaría de aprender.
- **Robustez a P₀** (10², 10⁴ y 10⁶·I): el burn-in es el mismo (908 días) y los pesos difieren como máximo **3.9·10⁻⁶** después del burn-in. **No depende de P₀.**
- **Innovaciones** (train, después del burn-in): var(ỹ/√S) = 1.07 y autocorrelación de orden 1 = 0.987. En el filtro 1 el spread es la innovación: esa autocorrelación es la reversión lenta (55.5 días) que se opera.

### 5.4 Ganancia de Kalman: cálculo e interpretación

K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹ es cuánto se mueve cada peso por cada unidad de sorpresa ỹ_t.
- Después del burn-in, |K| es del orden de 0.001 a 0.02 (figura de diagnóstico, escala lineal; puede ser negativa).
- **Ejemplo:** el 2014-08-13, ỹ = −0.0824 y K_NEM = −0.0198, así que ẑ_NEM sube (−0.0198)(−0.0824) = +0.0016, de 0.3303 a 0.3319.
- Con un Q tan chico, los pesos se mueven muy despacio y conservan la reversión.

### 5.5 Ejemplo numérico de 5 periodos (train, primeros días después del burn-in)

Recalculado a mano con la P completa. **Diferencia máxima contra el código: 0.**

Datos: μ = (NEM 3.27730, GFI 1.38020, FNV 3.90013, RGLD 4.10620); R = 0.014555; Q = diag(1.424·10⁻⁷, 5.707·10⁻⁸, 1.834·10⁻⁷); ẑ₀ = (0.4107, −0.5115, 0.5770).

y_t y C_t (log-precios centrados):

| Fecha | y_t (FNV) | NEM | GFI | RGLD |
|---|---|---|---|---|
| 2014-08-13 | 0.073007 | −0.234993 | −0.241050 | 0.156993 |
| 2014-08-14 | 0.026432 | −0.235732 | −0.233899 | 0.131830 |
| 2014-08-15 | 0.028324 | −0.241664 | −0.245846 | 0.114781 |
| 2014-08-18 | 0.033977 | −0.244271 | −0.238661 | 0.123216 |
| 2014-08-19 | 0.029011 | −0.243898 | −0.257938 | 0.120329 |

Predicción:

| Fecha | ẑ NEM | ẑ GFI | ẑ RGLD | P_{t\|t−1} NEM | P GFI | P RGLD |
|---|---|---|---|---|---|---|
| 2014-08-13 | 0.330291 | −0.435219 | 0.816060 | 0.001758 | 0.000563 | 0.000596 |
| 2014-08-14 | 0.331925 | −0.435967 | 0.815154 | 0.001752 | 0.000562 | 0.000594 |
| 2014-08-15 | 0.333939 | −0.436889 | 0.814094 | 0.001747 | 0.000561 | 0.000593 |
| 2014-08-18 | 0.335628 | −0.437651 | 0.813233 | 0.001742 | 0.000560 | 0.000592 |
| 2014-08-19 | 0.337352 | −0.438439 | 0.812350 | 0.001736 | 0.000559 | 0.000590 |

Innovación, su varianza y ganancia:

| Fecha | ỹ_t | S_t | K NEM | K GFI | K RGLD |
|---|---|---|---|---|---|
| 2014-08-13 | −0.082402 | 0.014616 | −0.019832 | 0.009085 | 0.010990 |
| 2014-08-14 | −0.104756 | 0.014611 | −0.019221 | 0.008799 | 0.010121 |
| 2014-08-15 | −0.091825 | 0.014606 | −0.018392 | 0.008295 | 0.009378 |
| 2014-08-18 | −0.088693 | 0.014611 | −0.019440 | 0.008890 | 0.009957 |
| 2014-08-19 | −0.099549 | 0.014605 | −0.018008 | 0.008033 | 0.009359 |

Corrección:

| Fecha | ẑ_{t\|t} NEM | GFI | RGLD | P_{t\|t} NEM | P GFI | P RGLD |
|---|---|---|---|---|---|---|
| 2014-08-13 | 0.331925 | −0.435967 | 0.815154 | 0.001752 | 0.000562 | 0.000594 |
| 2014-08-14 | 0.333939 | −0.436889 | 0.814094 | 0.001747 | 0.000561 | 0.000593 |
| 2014-08-15 | 0.335628 | −0.437651 | 0.813233 | 0.001742 | 0.000560 | 0.000591 |
| 2014-08-18 | 0.337352 | −0.438439 | 0.812350 | 0.001736 | 0.000558 | 0.000590 |
| 2014-08-19 | 0.339144 | −0.439239 | 0.811418 | 0.001732 | 0.000558 | 0.000589 |

**Cómo se lee:**
- Como A = I, ẑ_{t|t} de un día es ẑ_{t+1|t} del siguiente.
- ỹ es negativo los 5 días (FNV barata frente a la canasta): la desviación persiste, y esa es la reversión que se opera.
- P apenas baja porque ya está cerca de su piso.

### 5.6 Formulación de Powell (filtro 1)

**Proceso de 6 pasos:**

1. **Definir el problema:** estimar cada día los pesos que hacen estacionario al spread de NEM, GFI, FNV y RGLD, sin información futura y sin destruir la reversión.
2. **Modelar el estado:** z_t = β_t (3 pesos libres), con la creencia (ẑ_{t|t}, P_{t|t}).
3. **Modelar la transición:** z_t = I·z_{t−1} + W_t, con W_t ~ N(0, Q) y Q = q·Q_base.
4. **Modelar la observación:** y_t = C_t·z_t + V_t, con V_t ~ N(0, R), sobre log-precios centrados.
5. **Definir la regla de decisión:** publicar β̂_t = (+1, −ẑ_{t|t−1}); el spread del día es ỹ_t.
6. **Definir el mecanismo de aprendizaje:** las ecuaciones de actualización de Kalman (ganancia K_t), con q de la rejilla y R por OLS en train.

**Cinco elementos:**

| Elemento | Filtro 1 |
|---|---|
| Variables de estado | Creencia (ẑ_{t\|t}, P_{t\|t}); información: log-precios hasta t y μ de train; física: ninguna |
| Variables de decisión | β̂_{t+1} = (+1, −A·ẑ_{t\|t}) |
| Información exógena | Log-precios de t+1 (y_{t+1}, C_{t+1}) |
| Función de transición | Predicción y actualización de Kalman |
| Función objetivo | Minimizar E[Σ_t ‖z_t − ẑ_{t\|t−1}‖²] sujeto a un spread operable (ADF, vida media de 5 a 60 días, cero cambios de signo, pesos que se adaptan) |

## 6. Figuras (`artifacts/figures/`)

- **Figura 1** (`fig1_log_precios.png`): log-precios de los 4 activos del grupo final.
- **Figura 3** (`fig3_beta_t.png`): β̂_t por activo, con el β de Johansen de train de referencia.
- **Figura 4** (`fig4_eigenvector.png`): primer eigenvector móvil (solo datos pasados).
- **Convergencia** (`fig_diagnostico_filtro1.png`, train): P_t en escala log y |K_t| en escala **lineal**.

Las figuras 1, 3 y 4 tienen las fronteras train/test/valid dibujadas y etiquetadas. Se revisaron con vistas recortadas a train.

## 7. Limitaciones y declaraciones

1. **Correlación de 0.7 (etapa 2):** ningún subgrupo llega a 0.7 con retornos diarios (máximo 0.647 en energía y 0.652 en oro). El respaldo se adoptó **después** de ver el fracaso: es una **desviación declarada**.
2. **Data snooping y pruebas múltiples:**
   - **Se probaron dos sectores** (energía y oro), y el oro se eligió después de que energía fallara.
   - **Grupos evaluados sumando todas las corridas:**
     - en esta rama: 84 (energía) + 84 (oro) + 3 candidatos nuevos de 3, 4 y 5 activos = **171**;
     - con las versiones anteriores sobre el mismo train (2 × 8,008): **16,187**.
   - **Pruebas de hipótesis en esta rama:** 36 de Engle-Granger, **33 de Johansen** (30 del embudo más 3 candidatos nuevos) y 4 × 29 configuraciones de q.
   - **Falsos positivos esperados por azar al 5%:**
     - Engle-Granger: 36 × 0.05 = **1.8** (pasó 1);
     - Johansen: 33 × 0.05 = **1.65**;
     - si los 16,187 grupos se hubieran probado al 5%: alrededor de **809**.
   - **No sobrevive a Bonferroni:** con 33 pruebas de Johansen el umbral sería p < 0.0015 (más exigente que el 99%), y la traza del ganador **no pasa ni al 99%** (49.92 < 54.68).
   - Las correcciones formales para el desempeño resultante son el **Deflated Sharpe Ratio** (Bailey y López de Prado, 2014) y el **Reality Check** (White, 2000).
3. **Estabilidad baja:** 3 de 14 sub-ventanas (21%), ninguna después de 2015. Una ruptura fuera de muestra es probable.
4. **Burn-in del 40.1% de train** (908 días), por el q pequeño.
5. **ADF del spread de Kalman en el límite** (p = 0.0476), y el q elegido no viene de una meseta.
6. **Vida media cerca del máximo** (57.9 días con Johansen y 55.5 con Kalman).
7. **Train es en muestra** para el filtro.
8. Se excluyó Barrick por falta de datos.
