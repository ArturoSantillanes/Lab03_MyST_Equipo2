# Datos, selección del grupo por cointegración y filtro de Kalman 1 (Milca)

Todo lo que se estima, elige o calibra en esta sección usa **solo la partición de entrenamiento (train, 2011-01-03 a 2019-12-30)**. Cada cifra indica su tramo. El análisis trabaja con el **logaritmo natural** de los precios ajustados.

> **Notación.** En el filtro de Kalman, z_t es el estado oculto (los pesos β_t) y S_t es la varianza de la innovación. En Johansen, S_t = βᵀP_t es el spread. El z-score de las señales (Arturo) es otra cosa.

**Resumen (train):** se buscó en **8 sectores** con un protocolo y una regla de selección **registrados en git antes de descargar los datos nuevos** (commit `e52f6a2`). Ganó el grupo de **eléctricas reguladas DUK, XEL, ED, ES, CMS y AEE (6 activos)**, el único que, además de cumplir los 7 requisitos, queda primero en el orden de la regla. Sus cifras principales:
- correlación de retornos de 0.742 o más;
- r̂ = 1 por las dos pruebas de Johansen;
- ADF del spread p = 1.1·10⁻⁷ y vida media de 15.3 días;
- estabilidad de 11/14 (79%);
- meseta de q de 15 puntos contiguos y cero cambios de signo.

## 1. Universo de activos

### 1.1 Preregistro: sectores elegidos antes de mirar datos (`data/universe.json`)

La versión 4 (oro) no alcanzaba la correlación de 0.7 y tenía una estabilidad de 21%. **Antes de descargar un solo dato nuevo** se escribió y registró en git una lista de **7 sectores nuevos**, cada uno con su tendencia estocástica común. También se registraron, en el mismo archivo, sus candidatos con justificación por activo, el protocolo, la definición de meseta y la regla de selección. El oro se volvió a evaluar con el mismo protocolo. La única salvedad es el conocimiento general previo de qué sectores suelen tener retornos muy correlacionados, que es justo la razón para elegirlos.

| Sector | Tendencia común | Por qué se espera equilibrio de largo plazo |
|---|---|---|
| Oro (v1-v4) | Precio del oro | Valor ≈ onzas × precio del oro − costos; el mismo producto a un precio global |
| Bancos grandes | Ciclo de crédito de EE.UU., curva de tasas, regulación de la Fed | Mismos préstamos, misma curva de tasas y mismas pruebas de estrés |
| Bancos regionales | Tasas, depósitos y crédito comercial e inmobiliario | El mismo modelo de margen financiero; se mueven en bloque |
| Aseguradoras de daños | Ciclo de primas, catástrofes, rendimiento de bonos | El mismo producto en el mismo mercado |
| **Eléctricas reguladas** | **Tasas de largo plazo y rendimientos autorizados por los reguladores** | **Monopolios regulados: el regulador fija su rendimiento sobre la base de activos, así que se valúan casi como un mismo bono de largo plazo** |
| Petroleras integradas | Precio del crudo y márgenes de refinación | Todas integradas (a diferencia del intento 1 con productoras puras) |
| Semiconductores (equipo y analógicos) | Ciclo global de inventarios y de inversión | Mismos clientes y mismos mercados finales |
| Constructoras de vivienda | Tasas hipotecarias y demanda de vivienda nueva | El mismo producto en el mismo mercado |

**Excluidos de antemano, con razón escrita:**
- **Varias clases de acción o doble listado** (BRK-A/B, BF-A/B, HEI/HEI-A, LEN/LEN-B): forman grupos de 2 de la misma empresa (debajo del mínimo de 3) y su relación es contractual, no un equilibrio entre empresas.
- **ETF sustitutos:** son el mismo portafolio.
- **Aerolíneas:** no hay de 8 a 12 con historia desde 2011.
- **Retail:** formatos muy distintos.

**Regla de datos (preregistrada):** un ticker sin historia completa se excluye **sin reemplazo**. Se aplicó a BK, CMA, SNV y BZH: Yahoo ya no los tiene en 2026 (fusiones o cambio de ticker). Quedan **73 activos**: 9 de oro, 9 de bancos grandes, 8 regionales, 10 aseguradoras, 10 eléctricas, 9 petroleras, 10 de semiconductores y 8 constructoras.

### 1.2 Los 10 candidatos del sector ganador (justificación por activo)

La justificación de los 73 activos está en `data/universe.json`. Aquí van los del sector ganador:

| Ticker | Empresa | Segmento | Justificación |
|---|---|---|---|
| DUK | Duke Energy | Regulada (Carolinas, Florida, Medio Oeste) | Gran eléctrica regulada, valuada por su base de activos |
| SO | Southern Company | Regulada (Sureste) | Mismo modelo regulado que DUK |
| AEP | American Electric Power | Regulada y transmisión | Mismo motor de tasas |
| **XEL** | Xcel Energy | Regulada (Medio Oeste, Colorado) | Mismo modelo de rendimiento autorizado |
| **ED** | Consolidated Edison | Eléctrica y gas (Nueva York) | De muy bajo riesgo; casi un bono |
| WEC | WEC Energy | Eléctrica y gas (Wisconsin) | Mismo modelo que XEL y CMS |
| **ES** | Eversource | Distribución (Nueva Inglaterra) | Eléctrica regulada de distribución |
| **CMS** | CMS Energy | Eléctrica y gas (Michigan) | Mismo regulador estatal que DTE |
| DTE | DTE Energy | Eléctrica y gas (Michigan) | Mismo estado y regulador que CMS |
| **AEE** | Ameren | Regulada (Misuri, Illinois) | Mismo modelo de base de activos |

### 1.3 Historia de búsquedas (transparencia)

- **v1:** 9 productoras de petróleo (84 subgrupos). Ninguno pasó la correlación (máximo 0.647). Registro en `artifacts/cointegration/intento1_energia_etapa2.csv`.
- **Versiones anteriores del equipo:** 2 × 8,008 grupos de 16 petroleras sobre este mismo train.
- **v2-v4:** oro.
- **v5 (esta):** los 8 sectores del preregistro.

Todo se cuenta en la sección 7.

## 2. Datos (`data/`)

- **Fuente:** Yahoo Finance (yfinance 1.7.0), cierre ajustado por dividendos y splits. Cada sector se congeló en `data/raw/<sector>.csv` con su `_meta.json`. El oro conserva su descarga original (renombrada a `data/raw/oro.csv`).
- **Ventana:** 2011-01-01 a 2025-12-31, 15 años, el máximo permitido. Resultan 3,772 días hábiles (2011-01-03 a 2025-12-31).
- **`data/processed/prices.csv`** trae los **73 candidatos**. El grupo elegido está en `selected_group.json`, y `data/processed/sectors.json` dice a qué sector pertenece cada ticker.

| Caso | Encontrados | Regla de resolución |
|---|---|---|
| Ticker sin historia completa | 4 (BK, CMA, SNV, BZH) | Se excluye sin reemplazo |
| Valores faltantes / fechas desalineadas | 0 / 0 | La fecha se descarta para todos (no se rellena) |
| Fechas duplicadas | 0 | Se conserva la primera |
| Precios no positivos | 0 | Se tratan como faltantes |
| Saltos anómalos (\|log-retorno\| > 0.40) | 2 (MHO el 2020-03-18, FHN el 2023-05-04) | Se marcan sin modificar; ninguno revierte, así que no parecen error |

**Partición cronológica 60/20/20** (`data/processed/splits.json`):

| Tramo | Inicio | Fin | Días |
|---|---|---|---|
| train | 2011-01-03 | 2019-12-30 | 2,263 |
| test | 2019-12-31 | 2022-12-27 | 754 |
| valid | 2022-12-28 | 2025-12-31 | 755 |

## 3. Marco teórico

**Eigenvalores y eigenvectores.** Un eigenvector de una matriz es un vector que, al transformarse con esa matriz, **no cambia de dirección, solo de escala**. El factor de escala es su eigenvalor. Johansen calcula eigenvalores y eigenvectores de una matriz construida con los datos, de forma parecida a PCA: busca las combinaciones de activos más estables. **Los eigenvectores dicen cuánto invertir en cada activo.**

**Spread del portafolio.** S_t = βᵀ·P_t, con P_t el vector de **log-precios** de los n activos en t. β es el vector cuya combinación **revierte a la media**.

**Pruebas múltiples.** Mientras más activos y grupos se prueban, mayor la probabilidad de falsos positivos. Por eso cada sector se limita a 8-12 empresas con razón económica, y todo se cuenta (sección 7).

**Del VAR(p) al VECM.** Se parte de

P_t = A_1·P_{t−1} + A_2·P_{t−2} + … + A_p·P_{t−p} + ε_t

y se reescribe como modelo de corrección de error vectorial:

ΔP_t = Π·P_{t−1} + Σ_{i=1}^{p−1} Γ_i·ΔP_{t−i} + ε_t,   con Π = A_1 + … + A_p − I y Γ_i = −(A_{i+1} + … + A_p)

- A_i es la matriz de coeficientes: el aporte de cada rezago de cada activo.
- I es la matriz identidad; p es el número de rezagos del VAR.

Los casos desarrollados:
- **p = 1:** restando P_{t−1} de ambos lados, ΔP_t = (A_1 − I)·P_{t−1} + ε_t.
- **p = 2:** ΔP_t = (A_1 − I)·P_{t−1} + A_2·P_{t−2} + ε_t. Como P_{t−2} = P_{t−1} − ΔP_{t−1}:
  ΔP_t = (A_1 + A_2 − I)·P_{t−1} − A_2·ΔP_{t−1} + ε_t.
- **p = 3:** ΔP_t = (A_1 − I)·P_{t−1} + A_2·P_{t−2} + A_3·P_{t−3} + ε_t. Con P_{t−2} = P_{t−1} − ΔP_{t−1} y P_{t−3} = P_{t−1} − ΔP_{t−1} − ΔP_{t−2}:
  ΔP_t = (A_1 + A_2 + A_3 − I)·P_{t−1} − (A_2 + A_3)·ΔP_{t−1} − A_3·ΔP_{t−2} + ε_t.
  Es decir, Π = A_1 + A_2 + A_3 − I, Γ_1 = −(A_2 + A_3) y Γ_2 = −A_3.

**Estimación.** Por **máxima verosimilitud**: se busca la combinación de parámetros que mejor explica los datos observados. La **prueba de razón de verosimilitud (LR)** compara el modelo con r relaciones contra uno con más, y así determina r:
- **Traza:** H0 "hay a lo más r relaciones" contra "hay más de r".
- **Máximo eigenvalor:** H0 "hay r" contra "hay r + 1".

**Condición de rango.** El rango de Π es el número r de combinaciones estacionarias, y se escribe Π = αβᵀ: las r columnas de β son los vectores de cointegración y α es la velocidad de ajuste.
- **r̂ = 0 (alarma):** Π = 0, no hay ninguna combinación estacionaria. El grupo no sirve para reversión a la media.
- **r̂ = n (alarma):** Π tiene rango completo, así que cada serie sería estacionaria por sí sola. No hay tendencia común; casi siempre es un problema de datos o de especificación.
- **0 < r̂ < n:** hay r̂ relaciones de equilibrio y n − r̂ tendencias comunes. Es el caso que se busca.

## 4. Embudo de selección (7 etapas en cada sector, solo train, log-precios)

Se guarda una fila por subgrupo en `artifacts/cointegration/group_selection.csv` (938 filas, con sector, etapa alcanzada, estadísticos, decisión y razón).

**Etapa 1 (universo):** la sección 1. Todos los subgrupos de 6 activos de cada sector: C(n,6), es decir 84, 28 o 210 por sector, **938 en total**.

**Etapa 2 (correlación móvil de log-retornos; heurística):** ventana de 63 días; el criterio es una correlación media por pares > 0.70 en las 3 sub-ventanas de ~3 años.

**Etapa 3 (Engle-Granger por pares, p < 0.05; prioriza, no elimina):** da el orden de evaluación. Son 299 pares, calculados una vez por sector.

**Etapas 4 a 6:** se aplican a **todos** los subgrupos que pasan la etapa 2. Si en un sector ninguno la pasa, se aplican a todos y se cuenta como respaldo.
- Orden p del VAR por AIC y BIC (se usa BIC).
- **Constante sin restringir (caso 3), fuera de la relación de cointegración.** Valores críticos de **MacKinnon, Haug y Michelis (1996)**.
- r̂ con la traza al 95%; r̂ = 0 o r̂ = n se rechazan.
- **Etapa 5:** spread de cada columna de β; se queda la de ADF más negativo con vida media operable.
- **Etapa 6:** ADF p < 0.05 y vida media de 5 a 60 días.

**Etapa 7 (estabilidad):** se repiten las etapas 4 a 6 en 14 sub-ventanas de 504 días que avanzan 126, para **todo** subgrupo que llega a Johansen. Una sub-ventana es estable si r̂ ≥ 1 por la traza **y** el spread de la mejor columna dentro de r̂ pasa ADF.

| Sector | Subgrupos | Pasan correlación (máx.) | Respaldo | Llegan a Johansen | r̂ = 0 | Cumplen 4-6 | Estabilidad máx. |
|---|---|---|---|---|---|---|---|
| Oro | 84 | 0 (0.652) | sí | 84 | 77 | 6 | 43% |
| Bancos grandes | 84 | 47 (0.749) | no | 47 | 47 | 0 | 29% |
| Bancos regionales | 28 | 27 (0.733) | no | 27 | 13 | 14 | 50% |
| Aseguradoras | 210 | 0 (0.640) | sí | 210 | 207 | 2 | 64% |
| **Eléctricas** | 210 | **210 (0.780)** | no | 210 | 165 | **44** | **86%** |
| Petroleras integradas | 84 | 0 (0.697) | sí | 84 | 80 | 4 | 43% |
| Semiconductores | 210 | 0 (0.658) | sí | 210 | 197 | 13 | 21% |
| Constructoras | 28 | 1 (0.723) | no | 1 | 1 | 0 | 7% |
| **Total** | **938** | 285 | 4 sectores | **873** | 787 | **83** | |

- No apareció ninguna alarma de r̂ = n.
- **El respaldo de la etapa 2 se usó en 4 sectores** (oro, aseguradoras, petroleras y semiconductores): **contador de respaldos = 4**. Ninguno de esos sectores puede ganar, porque R1 exige pasar la correlación.

### 4.8 Recorte de patas y selección final (solo train)

**Recorte (preregistrado):** a cada subgrupo que cumple 4 a 6 se le quitan las patas con |β_i| / Σ|β| < **5% del nocional**, dejando al menos 3 activos, y el recorte se evalúa completo.
- Con 6 activos, el peso igual sería 16.7%.
- Con el 80% de $1,000,000, el 5% son $40,000, monto en el que la comisión anula el aporte de la pata.
- El enunciado permite grupos de 2 a 8 activos.

Se generaron **37 recortes**. Junto con los 873 subgrupos, se evaluaron completos **910 grupos**. Cada uno con Johansen, columnas, estabilidad y, si tiene vector válido, el filtro 1 con la rejilla fina de q (141 valores, más una ultrafina cuando hay q válidos sin meseta).

**Regla de selección (preregistrada en `data/universe.json` antes de ver datos):** un grupo es **elegible** si cumple los siete requisitos.

| | Requisito |
|---|---|
| R1 | Correlación media por pares de retornos > 0.7 en las 3 sub-ventanas |
| R2 | r̂ ≥ 1 (y < n) por la traza **y** por el máximo eigenvalor al 95% |
| R3 | ADF del spread de Johansen p < 0.05 |
| R4 | Vida media de 5 a 60 días |
| R5 | Existe meseta de q (≥ 5 q válidos contiguos en la rejilla fina) con cero cambios de signo |
| R6 | ADF del spread de Kalman p < 0.05 en el q elegido |
| R7 | Estabilidad ≥ 50% de las sub-ventanas |

- **Orden entre elegibles:** mayor estabilidad, luego ADF más negativo, luego menor burn-in (% de train), luego más activos.
- **Si no hubiera ninguno elegible:** no se baja nada. Se ordena por número de requisitos cumplidos y se declara qué falla.
- **R3 y R4** se miden en la columna elegida. Si no hay, se miden en la de ADF más negativo dentro de r̂, solo para contar.

**Cuántos cumplen cada requisito (de 910):** R1 311, R2 76, R3 902, R4 716, **R5 29**, R6 35, **R7 77**. Los cuellos de botella son la meseta de q y la estabilidad.

**Requisitos cumplidos (de 910):** 7/7: **4** · 6/7: 19 · 5/7: 18 · 4/7: 82 · 3/7: 210 · ≤ 2/7: 577.

**Los 12 primeros (tabla completa en `artifacts/cointegration/candidatos_final.csv`):**

| # | Grupo | R1 | R2 | R3 | R4 | R5 | R6 | R7 | Cumple | Estabilidad | ADF | Burn-in |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **DUK-XEL-ED-ES-CMS-AEE** | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | **7** | **79%** | −6.08 | 37% |
| 2 | XEL-ED-ES-CMS-AEE (recorte) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7 | 71% | −6.36 | 35% |
| 3 | XEL-ED-ES-CMS-DTE-AEE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7 | 57% | −5.74 | 33% |
| 4 | SO-AEP-XEL-WEC-ES-AEE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | 7 | 50% | −5.34 | 33% |
| 5 | SO-XEL-WEC-ES-CMS-AEE | ✓ | ✓ | ✓ | ✓ | ✗ (4) | ✓ | ✓ | 6 | 64% | −5.54 | 42% |
| 6 | AEP-XEL-ED-WEC-ES-CMS | ✓ | ✗ (máx. eig. 0) | ✓ | ✓ | ✓ | ✓ | ✓ | 6 | 64% | −5.47 | 30% |
| 7 | DUK-AEP-XEL-ED-ES-AEE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 43% | −5.88 | 38% |
| 8 | DUK-XEL-ED-WEC-ES-AEE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 43% | −5.75 | 36% |
| 9 | SO-AEP-XEL-ED-WEC-ES | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 43% | −5.52 | 22% |
| 10 | AEP-XEL-ED-ES (recorte) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 43% | −5.42 | 23% |
| 11 | SO-XEL-WEC-ES-AEE (recorte) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 43% | −5.18 | 42% |
| 12 | AEP-XEL-ED-WEC-ES-AEE | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | 6 | 36% | −6.12 | 36% |

**Mejor grupo de cada sector:**

| Sector | Mejor grupo | Cumple | Qué falla |
|---|---|---|---|
| Eléctricas | DUK-XEL-ED-ES-CMS-AEE | 7 | — |
| Bancos regionales | KEY-FITB-HBAN-TFC (recorte) | 6 | R7 (29%) |
| Oro | NEM-GFI-FNV-RGLD (ganador de v4) | 4 | R1 (0.587), R5 (racha de 4), R7 (21%) |
| Petroleras integradas | BP-SHEL-E-EQNR-SU-IMO | 3 | R1 (0.640), R5, R6 (sin q válido), R7 (36%) |
| Bancos grandes | JPM-BAC-WFC-USB-PNC-MS | 3 | R2 (r̂ = 0), R5, R6, R7 (29%) |
| Semiconductores | AMAT-KLAC-MCHP-QCOM (recorte) | 3 | R1 (0.518), R5, R6, R7 (21%) |
| Aseguradoras | ALL-PGR-HIG-CINF-L-AFG | 3 | R1 (0.573), R5, R6, R7 (14%) |
| Constructoras | DHI-LEN-PHM-TOL-KBH-MTH | 3 | R2 (r̂ = 0), R5, R6, R7 (7%) |

**Ganador: DUK, XEL, ED, ES, CMS y AEE (6 activos).** Es uno de los 4 elegibles y el de mayor estabilidad (79%).
- Llegó desde el lugar 143 de 210 en la prioridad de Engle-Granger: un solo par de 15 pasa EG. Eso muestra que EG solo ordena y que la regla anterior, que se detenía en el primero que cumplía, habría dejado mejores grupos sin probar.
- **Nota sobre CMS:** pesa **4.7% del nocional**, debajo del 5%. La regla no excluye grupos por eso. Su recorte (DUK-XEL-ED-ES-AEE) se evaluó completo y **baja a 29% de estabilidad (no cumple R7)**: la pata pequeña sí aporta estabilidad.
- El segundo elegible (XEL-ED-ES-CMS-AEE) tiene todas sus patas arriba del 10%, pero menor estabilidad (71%). Se reporta como alternativa; la regla no se cambió.

### 4.9 Grupo ganador: Johansen, vector y spread (train)

- p = 1 (AIC = BIC = 1); constante sin restringir; valores críticos de MacKinnon, Haug y Michelis.

| Hipótesis | Traza | VC 95% | VC 99% | Máx. eigenvalor | VC 95% | VC 99% |
|---|---|---|---|---|---|---|
| r = 0 | **101.56** | 95.75 | 104.96 | **51.57** | 40.08 | 45.87 |
| r ≤ 1 | 49.99 | 69.82 | 77.82 | 26.51 | 33.88 | 39.37 |
| r ≤ 2 | 23.48 | 47.85 | 54.68 | 11.41 | 27.59 | 32.72 |
| r ≤ 3 | 12.07 | 29.80 | 35.46 | 8.86 | 21.13 | 25.87 |
| r ≤ 4 | 3.22 | 15.49 | 19.93 | 2.87 | 14.26 | 18.52 |
| r ≤ 5 | 0.35 | 3.84 | 6.63 | 0.35 | 3.84 | 6.63 |

- **r̂ = 1, con las dos pruebas de acuerdo**, y sin alarma (ni 0 ni n).
- La traza pasa al 95% (+6% sobre el valor crítico) pero **no al 99%**. El máximo eigenvalor sí pasa al 99% (+12%).
- **Eigenvalores:** 0.02254, 0.01165, 0.00503, 0.00391, 0.00127, 0.00015.
- **Correlación de retornos por sub-ventana:** 0.742, 0.840 y 0.838. La de log-precios en niveles es 0.983, solo como dato informativo.
- **Etapa 5:** columna 0 (ADF −6.08, vida media 15.3). Las demás tienen ADF de −1.3 a −3.0 y vidas medias de 56 a 375 días.
- **Normalización:** la pata de mayor |β| es **XEL** (+1).

| | DUK | XEL | ED | ES | CMS | AEE |
|---|---|---|---|---|---|---|
| β (train) | +0.186 | **+1** | −0.484 | −0.658 | +0.131 | −0.321 |
| \|β\| | 0.186 | 1.000 | 0.484 | 0.658 | 0.131 | 0.321 |
| Participación en el nocional | 6.7% | 36.0% | 17.4% | 23.7% | **4.7%** | 11.5% |

- **Convención de signo:** largo el spread = comprar XEL, DUK y CMS (β > 0) y vender ES, ED y AEE (β < 0), en proporción a |β|. Corto el spread invierte todos los signos.
- **Etapa 6:** ADF −6.08, **p = 1.1·10⁻⁷**; vida media de **15.3 días**.
- **Etapa 7:** **estable en 11 de 14 sub-ventanas (79%)**.

| Sub-ventana | r̂ | ADF p | Vida media | Estable |
|---|---|---|---|---|
| 2011-01 a 2013-01 | 2 | 5.6e-5 | 7.5 | sí |
| 2011-07 a 2013-07 | 1 | 6.2e-6 | 5.6 | sí |
| 2012-01 a 2014-01 | 1 | 7.5e-7 | 4.1 | sí |
| 2012-07 a 2014-07 | 1 | 9.5e-8 | 4.5 | sí |
| 2013-01 a 2015-01 | 1 | 0.0027 | 8.0 | sí |
| 2013-07 a 2015-07 | 1 | 1.2e-6 | 5.5 | sí |
| 2014-01 a 2016-01 | 1 | 2.0e-9 | 3.7 | sí |
| 2014-07 a 2016-07 | 1 | 2.8e-10 | 3.4 | sí |
| 2015-01 a 2017-01 | 1 | 1.0e-4 | 2.9 | sí |
| 2015-07 a 2017-07 | 0 | 0.0012 | 23.2 | no |
| 2016-01 a 2018-01 | 1 | 0.027 | 6.3 | sí |
| 2016-07 a 2018-07 | 1 | 3.6e-5 | 6.8 | sí |
| 2017-01 a 2019-01 | 0 | 2.2e-5 | 7.0 | no |
| 2017-07 a 2019-07 | 0 | 2.3e-5 | 6.9 | no |

Las 3 sub-ventanas inestables están al final de train (r̂ = 0 por la traza, aunque el spread siga pasando ADF). Además, las vidas medias en ventanas de 2 años son más cortas (3 a 8 días) que en todo train (15.3): el vector de cada ventana se ajusta mejor dentro de su muestra.

**Pérdida de cointegración fuera de muestra.** Si la cointegración se pierde fuera de la muestra, la estrategia pierde sentido. Aquí se **minimiza el error dentro de la muestra (train)**. Que las sub-ventanas inestables estén al final de train es una señal de alerta de ruptura posterior. Se documenta en vez de esconderse.

## 5. Filtro de Kalman 1 (`kalman/filter1_weights.py`)

### 5.1 Modelo (notación del profesor)

- **Estado:** z_t = β_t, los pesos de DUK, ED, ES, CMS y AEE frente a XEL. **Sin intercepto α.**
- **Transición:** z_t = A·z_{t−1} + W_t, W_t ~ N(0, Q), con **A = I** explícita. Q es la velocidad del drift de los pesos.
- **Observación:** y_t = C_t·z_t + V_t, V_t ~ N(0, R).
  - y_t = log P_XEL − μ_XEL; C_t = (log P_i − μ_i) de las otras cinco.
  - μ = media de train de cada log-precio: es el término determinístico, consistente con Johansen caso 3.
- **Predecir:** ẑ_{t|t−1} = A·ẑ_{t−1|t−1}; P_{t|t−1} = A·P_{t−1|t−1}·Aᵀ + Q.
- **Actualizar:** ỹ_t = y_t − C_t·ẑ_{t|t−1}; S_t = C_t·P_{t|t−1}·C_tᵀ + R; K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹; ẑ_{t|t} = ẑ_{t|t−1} + K_t·ỹ_t; P_{t|t} = (I − K_t·C_t)·P_{t|t−1}.
- **Pesos β̂_t:** +1 en XEL y −ẑ_{t|t−1} en las demás.
- **Spread** (sin look-ahead, con los pesos previos): β̂_tᵀ(P_t − μ) = ỹ_t.

**Aclaración sobre la regla de no mirar test ni valid:**
- El filtro se **corre** hacia adelante sobre toda la muestra con parámetros **congelados en train**, porque Arturo y Paula necesitan esas series. Lo mismo vale para el eigenvector móvil (solo datos pasados en cada fecha).
- **Nada se estima, calibra ni elige con test o valid**, y ninguna cifra de esos tramos se usó para decidir.
- **Eso no es look-ahead.**

### 5.2 Selección de Q y R (train)

- **R** = varianza residual de la regresión OLS estática de y sobre C (sin constante, log-precios centrados): **R = 2.81·10⁻⁴** (√R = 0.0168). Es un ruido de alrededor de **1.7% en el log-precio de XEL** que la canasta no explica.
- **Q** = q·Q_base, con Q_base = R·diag(1/E[C_i²]): por unidad de q, cada componente aporta lo mismo a la varianza de la predicción.
- **No se usó máxima verosimilitud libre:** elige un Q demasiado grande, los pesos persiguen cada desviación y el filtro se come la reversión.
- **Criterio, escrito antes:**
  - (a) ADF del spread de Kalman p < 0.05;
  - (b) vida media de 5 a 60 días;
  - (c) **cero** cambios de signo en train, después del burn-in;
  - (d) los pesos se adaptan: al final de train difieren al menos 0.05 de los de Q = 0.

**Fase A: buscar la meseta.**
- La rejilla de v4 tenía 29 valores (4 por década). Ahora es **fina: 141 valores de 10⁻⁸ a 10⁻¹ (20 por década)**.
- **Meseta** (definida de antemano) = ≥ 5 q válidos **contiguos**: q puede moverse ±26% alrededor del centro sin salir de la región válida.
- Si hay q válidos sin meseta, se refina a 100 por década para concluir.

| Grupo (train) | Configuraciones | q válidos | Racha contigua más larga | ¿Meseta? |
|---|---|---|---|---|
| NEM-GFI-FNV-RGLD (oro, v4) | 141 finas + 61 ultrafinas | 5 finas; 28 ultrafinas | **4** finas (2.2·10⁻⁶ a 3.2·10⁻⁶); 19 ultrafinas (0.18 décadas) | **No**: le falta 1 punto; los demás q válidos son aislados (p del ADF oscila alrededor de 0.05) |
| **DUK-XEL-ED-ES-CMS-AEE (ganador)** | **141** | **25** | **15** (5.0·10⁻⁶ a 2.5·10⁻⁵, 0.7 décadas) | **Sí** |

**Detalle de la meseta del ganador** (`artifacts/kalman/filter1_q_grid.csv`; figura `fig_rejilla_q.png`):

| q | Burn-in (días) | ADF p (Kalman) | Vida media | Adaptación | Cambios de signo |
|---|---|---|---|---|---|
| 5.0·10⁻⁶ (inicio de la meseta) | 828 | 0.020 | 22.6 | 0.128 | 0 |
| 6.3·10⁻⁶ | 860 | 0.04999993 (justo debajo) | 22.3 | 0.142 | 0 |
| 1.0·10⁻⁵ | 852 | 0.039 | 20.8 | 0.172 | 0 |
| **1.12·10⁻⁵ (centro, elegido)** | **840** | **0.0136** | **20.4** | **0.179** | **0** |
| 1.6·10⁻⁵ | 876 | 0.011 | 19.4 | 0.202 | 0 |
| 2.5·10⁻⁵ (fin de la meseta) | 941 | 0.030 | 17.3 | 0.232 | 0 |
| 2.8·10⁻⁵ (fuera) | 945 | 0.028 | – | – | **2** |

- **Cumplen 25 de 141** en dos regiones: 7.1·10⁻⁷ a 2.0·10⁻⁶ (10 puntos) y 5.0·10⁻⁶ a 2.5·10⁻⁵ (15 puntos).
- La racha más larga tiene 15 puntos ≥ 5, así que **hay meseta**. **q = 1.12·10⁻⁵ es su centro**, no está en la orilla de la rejilla y Q/R = 1.14·10⁻⁴.
- Si el punto de p = 0.0500 cayera del otro lado, la meseta seguiría existiendo: quedaría una racha de 12.
- Justo arriba de la meseta aparecen cambios de signo (2 en q = 2.8·10⁻⁵ y 4 en 3.2·10⁻⁵: falla el criterio c). Entre las dos regiones, de 2.2·10⁻⁶ a 4.5·10⁻⁶, el ADF da p de 0.052 a 0.065 (falla el criterio a).
- **Cero cambios de signo** (DUK 0, ED 0, ES 0, CMS 0, AEE 0).
- **Diagonal de Q:** DUK 5.15·10⁻⁸; ED 4.37·10⁻⁸; ES 2.86·10⁻⁸; CMS 1.70·10⁻⁸; AEE 1.98·10⁻⁸.
- **No hubo respaldos en q** (contador: 0).

### 5.3 Inicialización, burn-in y convergencia (train)

- **P₀ = 10⁴·I** (difuso: los |β| normalizados son ≤ 1). **ẑ₀** = −β de Johansen de train para las 5 patas libres.
- **Burn-in medido = 840 días (hasta 2014-05-07): el 37.1% de train.**
  - Definición: primer día desde el cual la varianza posterior de todos los componentes se queda bajo el techo de su banda estable, exp(mediana ± 3·MAD) del log de P en train, hasta el final de train.
- **Convergencia:** P_t baja de 10⁴ y se estabiliza.
  - **Piso** (mediana después del burn-in): DUK 1.6·10⁻⁴; ED 1.1·10⁻⁴; ES 4.2·10⁻⁴; CMS 2.3·10⁻⁴; AEE 1.3·10⁻⁴. Mínimo en train: 9.7·10⁻⁵.
  - **Con Q > 0, P converge a un piso mayor que cero**: cada día Q le suma incertidumbre (los pesos verdaderos sí se mueven) y la observación se la quita. El piso es donde las dos cosas se equilibran. Solo con Q = 0 llegaría a cero, y el filtro dejaría de aprender.
- **Robustez a P₀** (10², 10⁴ y 10⁶·I):
  - el burn-in es el mismo (840 días);
  - después del burn-in, los pesos difieren como máximo **8.3·10⁻⁶** y el spread **4.8·10⁻⁷**.
  - **No depende de P₀.**
- **Innovaciones** (train, después del burn-in): var(ỹ/√S) = 0.79 y autocorrelación de orden 1 = 0.967 (Ljung-Box p ≈ 0).
  - En el filtro 1 la innovación **es el spread**, y esa autocorrelación es la reversión lenta (20.4 días) que se opera. Por eso no es blanca, y no debe serlo.
  - La prueba de innovaciones blancas del enunciado aplica al filtro 2 (Arturo).

**Signo de CMS (importante para Arturo y Paula):**
- Con el β de Johansen, CMS vale +0.131. Con el filtro, después del burn-in, vale **alrededor de −0.29 en todo train**, sin cambiar de signo.
- El filtro reestima la relación desde el inicio (P₀ difuso) y, para una pata tan pequeña, el signo no está bien identificado.
- **Las posiciones deben usar β̂_t de `beta_t.csv`**, no el β estático.

### 5.4 Ganancia de Kalman: cálculo e interpretación

K_t = P_{t|t−1}·C_tᵀ·S_t⁻¹ es cuánto se mueve cada peso por cada unidad de sorpresa ỹ_t.
- Durante el burn-in, |K| llega a cientos (P es enorme y el filtro aprende rápido).
- Después, |K| baja al orden de 0.01 a 0.05 (figura de diagnóstico, escala **lineal**; K puede ser negativa).
- **Ejemplo:** el 2014-05-07, ỹ = 0.006393 y K_DUK = 0.052597, así que ẑ_DUK cambia (0.052597)(0.006393) = +0.000336, de −0.204080 a −0.203744.
- Con un Q chico, los pesos se mueven despacio y conservan la reversión.

### 5.5 Ejemplo numérico de 5 periodos (train, primeros días después del burn-in)

Recalculado a mano con numpy, escribiendo las ecuaciones sin llamar a `kalman_filter`, y comparado con `beta_t.csv`, `spread_t.csv`, `gain_1.csv` y `posvar_1.csv`. **Diferencia máxima: 0.**

**Datos:**
- μ = (DUK 3.84447, XEL 3.23122, ED 3.79245, ES 3.51149, CMS 3.19387, AEE 3.44647);
- R = 2.80817·10⁻⁴;
- Q = diag(5.148, 4.375, 2.856, 1.702, 1.982)·10⁻⁸ (DUK, ED, ES, CMS, AEE).

y_t y C_t (log-precios centrados):

| Fecha | y_t (XEL) | DUK | ED | ES | CMS | AEE |
|---|---|---|---|---|---|---|
| 2014-05-07 | −0.174159 | −0.054777 | −0.193140 | −0.097173 | −0.154345 | −0.119104 |
| 2014-05-08 | −0.188484 | −0.068142 | −0.199911 | −0.108514 | −0.166982 | −0.135836 |
| 2014-05-09 | −0.202692 | −0.080019 | −0.221213 | −0.123692 | −0.182837 | −0.160709 |
| 2014-05-12 | −0.211839 | −0.083080 | −0.233161 | −0.137331 | −0.195498 | −0.175057 |
| 2014-05-13 | −0.208562 | −0.083637 | −0.233889 | −0.133132 | −0.193090 | −0.173510 |

Predicción:

| Fecha | ẑ DUK | ẑ ED | ẑ ES | ẑ CMS | ẑ AEE | diag P_{t\|t−1} (DUK, ED, ES, CMS, AEE) |
|---|---|---|---|---|---|---|
| 2014-05-07 | −0.204080 | 0.475316 | 0.108411 | 0.371111 | 0.269637 | 5.65, 1.52, 8.85, 7.88, 2.24 (·10⁻⁴) |
| 2014-05-08 | −0.203744 | 0.475029 | 0.108657 | 0.370816 | 0.269744 | 5.65, 1.52, 8.85, 7.88, 2.24 |
| 2014-05-09 | −0.203632 | 0.474906 | 0.108749 | 0.370712 | 0.269787 | 5.64, 1.51, 8.85, 7.87, 2.24 |
| 2014-05-12 | −0.203284 | 0.474422 | 0.108961 | 0.370516 | 0.269895 | 5.64, 1.51, 8.84, 7.87, 2.24 |
| 2014-05-13 | −0.202659 | 0.473635 | 0.109096 | 0.370354 | 0.270036 | 5.64, 1.50, 8.85, 7.87, 2.24 |

Innovación, su varianza y ganancia:

| Fecha | ỹ_t (spread) | S_t | K DUK | K ED | K ES | K CMS | K AEE |
|---|---|---|---|---|---|---|---|
| 2014-05-07 | 0.006393 | 0.000283 | 0.052597 | −0.044948 | 0.038400 | −0.046138 | 0.016675 |
| 2014-05-08 | 0.002948 | 0.000283 | 0.038075 | −0.041868 | 0.031507 | −0.035358 | 0.014727 |
| 2014-05-09 | 0.010658 | 0.000283 | 0.032593 | −0.045339 | 0.019810 | −0.018383 | 0.010139 |
| 2014-05-12 | 0.016535 | 0.000283 | 0.037823 | −0.047618 | 0.008163 | −0.009800 | 0.008481 |
| 2014-05-13 | 0.018155 | 0.000283 | 0.035737 | −0.047602 | 0.014349 | −0.012946 | 0.008112 |

Corrección:

| Fecha | ẑ_{t\|t} DUK | ED | ES | CMS | AEE | diag P_{t\|t} (·10⁻⁴) |
|---|---|---|---|---|---|---|
| 2014-05-07 | −0.203744 | 0.475029 | 0.108657 | 0.370816 | 0.269744 | 5.65, 1.52, 8.85, 7.88, 2.24 |
| 2014-05-08 | −0.203632 | 0.474906 | 0.108749 | 0.370712 | 0.269787 | 5.64, 1.51, 8.85, 7.87, 2.24 |
| 2014-05-09 | −0.203284 | 0.474422 | 0.108961 | 0.370516 | 0.269895 | 5.64, 1.51, 8.84, 7.87, 2.24 |
| 2014-05-12 | −0.202659 | 0.473635 | 0.109096 | 0.370354 | 0.270036 | 5.64, 1.50, 8.84, 7.87, 2.24 |
| 2014-05-13 | −0.202010 | 0.472771 | 0.109356 | 0.370119 | 0.270183 | 5.63, 1.50, 8.84, 7.87, 2.24 |

**Cómo se lee:**
- Como A = I, ẑ_{t|t} de un día es ẑ_{t+1|t} del siguiente.
- ỹ es positivo los 5 días y crece: XEL está cara frente a la canasta.
- Cada peso se mueve muy poco (K·ỹ ≈ 10⁻⁴) y P apenas baja, porque ya está en su piso.
- Pesos publicados el 2014-05-07: β̂ = (+1 XEL, +0.204 DUK, −0.475 ED, −0.108 ES, −0.371 CMS, −0.270 AEE).

### 5.6 Formulación de Powell (filtro 1)

**Proceso de 6 pasos:**
1. **Definir el problema:** estimar cada día los pesos que hacen estacionario al spread de las 6 eléctricas, sin información futura y sin destruir la reversión.
2. **Modelar el estado:** z_t = β_t (5 pesos libres), con la creencia (ẑ_{t|t}, P_{t|t}).
3. **Modelar la transición:** z_t = I·z_{t−1} + W_t, con W_t ~ N(0, Q) y Q = q·Q_base.
4. **Modelar la observación:** y_t = C_t·z_t + V_t, con V_t ~ N(0, R), sobre log-precios centrados.
5. **Definir la regla de decisión:** publicar β̂_t = (+1, −ẑ_{t|t−1}); el spread del día es ỹ_t.
6. **Definir el mecanismo de aprendizaje:** las ecuaciones de actualización de Kalman (ganancia K_t), con q en el centro de la meseta y R por OLS en train.

**Cinco elementos:**

| Elemento | Filtro 1 |
|---|---|
| Variables de estado | Creencia (ẑ_{t\|t}, P_{t\|t}); información: log-precios hasta t y μ de train; física: ninguna |
| Variables de decisión | β̂_{t+1} = (+1, −A·ẑ_{t\|t}) |
| Información exógena | Log-precios de t+1 (y_{t+1}, C_{t+1}) |
| Función de transición | Predicción y actualización de Kalman |
| Función objetivo | Minimizar E[Σ_t ‖z_t − ẑ_{t\|t−1}‖²] sujeto a un spread operable (ADF, vida media de 5 a 60 días, cero cambios de signo, pesos que se adaptan) |

## 6. Figuras (`artifacts/figures/`)

- **Figura 1** (`fig1_log_precios.png`): log-precios de los 6 activos del grupo final.
- **Figura 3** (`fig3_beta_t.png`): β̂_t por activo, con el β de Johansen de train de referencia.
- **Figura 4** (`fig4_eigenvector.png`): primer eigenvector de Johansen móvil (504 días pasados, vector unitario).
- **Convergencia** (`fig_diagnostico_filtro1.png`): P_t en escala log y |K_t| en escala **lineal**, con la banda estable de train y el fin del burn-in.
- **Rejilla de q** (`fig_rejilla_q.png`, train): los criterios (a)-(d) en función de q, la meseta sombreada y el q elegido.

Las figuras 1, 3, 4 y la de convergencia tienen las fronteras train/test/valid dibujadas y etiquetadas. La de la rejilla no es una serie de tiempo; su título dice que es de train.

## 7. Limitaciones y declaraciones

1. **Data snooping y pruebas múltiples (conteo exacto, `selected_group.json` → `multiple_testing`):**
   - **v5:** 8 sectores, 73 activos, 938 subgrupos en la etapa 2 más 37 recortes, o sea **975 grupos**. Pruebas:
     - **299** de Engle-Granger;
     - **910** de Johansen en train;
     - **12,740** de Johansen en sub-ventanas;
     - **81,195** ADF sobre columnas de Johansen;
     - **16,527** configuraciones de q, cada una con su ADF.
   - **Con todas las versiones anteriores:** 16,187 + 975 = **17,162 grupos**, y **10 sectores o universos** distintos: energía v1, petroleras del equipo, oro y 7 sectores nuevos.
   - **Falsos positivos esperados por azar al 5%:** 910 × 0.05 = **45.5** pruebas de Johansen "significativas" aunque nada estuviera cointegrado.
   - **Bonferroni** para 910 pruebas de Johansen: p < 5.5·10⁻⁵. La traza del ganador **no pasa ni al 99%**, así que **no sobrevive a Bonferroni**. El ADF de su spread (p = 1.1·10⁻⁷) sí quedaría debajo de ese umbral, pero ese p usa valores críticos de Dickey-Fuller para un vector **estimado**, así que es optimista.
   - **Lo que defiende al ganador frente al azar:**
     - pasa además 6 requisitos independientes de la prueba de rango, entre ellos 79% de estabilidad, una meseta de q y la correlación de retornos;
     - 21 de los 23 grupos con 6 o más requisitos son del mismo sector (eléctricas): la relación aparece de forma consistente en el sector y no en un solo grupo afortunado.
   - Las correcciones formales para el desempeño resultante son el **Deflated Sharpe Ratio** (Bailey y López de Prado, 2014) y el **Reality Check** (White, 2000); le tocan al backtest.
2. **Se buscó entre sectores:** ampliar la búsqueda mejora la evidencia, pero es un grado de libertad del investigador. Se mitigó registrando la lista y la regla antes de ver datos (commit `e52f6a2`).
3. **Respaldo de la etapa 2 en 4 sectores** (contador = 4). No afecta al ganador, que pasa la correlación.
4. **Burn-in del 37.1% de train** (840 días), por el q pequeño.
5. **La traza no pasa al 99%** (101.56 contra 104.96).
6. **Estabilidad del 79%:** las 3 sub-ventanas inestables son las últimas de train.
7. **CMS pesa 4.7% del nocional** y su peso de Kalman tiene signo contrario al de Johansen (estable después del burn-in).
8. **Innovaciones del filtro 1 autocorrelacionadas** (0.967): es la reversión del spread, no un defecto del filtro.
9. **Train es en muestra** para el filtro.
10. **Exclusiones de datos:** BK, CMA, SNV y BZH (sin reemplazo); Barrick en el oro (v1).
