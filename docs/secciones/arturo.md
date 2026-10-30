# Filtro de Kalman 2, VECM, z-score y señales (Arturo)

Código: `kalman/filter2_spread.py`, `strategy/vecm.py`, `strategy/zscore.py`, `strategy/signals.py` y `strategy/figures.py`. Pruebas: `tests/test_causality.py`, `test_innovations.py`, `test_signals.py` y `test_filter2_step.py`.

Toda cifra indica su tramo. Train va del 2011-01-03 al 2019-12-30, test del 2019-12-31 al 2022-12-27 y valid del 2022-12-28 al 2025-12-31.

Lo que recibo de Milca:
- el spread S_t = β̂_tᵀ(log P_t − μ_train) del filtro 1;
- los pesos β̂_t (XEL = +1);
- la vida media de Johansen de 15.3 días (train);
- el burn-in del filtro 1 hasta el 2014-05-07.

Todo lo que estimo o elijo usa solo train. Los filtros se corren hacia adelante en test y valid con parámetros congelados.

## 1. Filtro de Kalman 2 (`kalman/filter2_spread.py`)

### 1.1 Qué hace, en palabras simples

El filtro 1 entrega cada día un número, el spread S_t. Para decidir si S_t está "lejos" de lo normal hacen falta dos cosas:
- **dónde está lo normal:** la media μ_t;
- **qué tanto se mueve normalmente:** la varianza σ_t².

El filtro 2 aprende ambas en tiempo real, un día a la vez. Con ellas calcula z_t = (S_t − μ̂_t)/σ̂_t.

### 1.2 Modelo (notación del profesor, A = 1)

El spread revierte a una media que se mueve despacio:

S_t − μ_t = Σ_{i=1}^{p} φ_i·(S_{t−i} − μ_{t−i}) + ε_t

| Pieza | Media latente μ_t | Varianza latente σ_t² |
|---|---|---|
| Transición | μ_t = μ_{t−1} + W^μ_t, W^μ ~ N(0, Q_μ) | σ²_t = σ²_{t−1} + W^σ_t, W^σ ~ N(0, Q_σ) |
| Observación | y_t = S_t − Σφ_i·S_{t−i} = C_μ·μ_t + V_t | u_t = ỹ_t² − C_μ²·P^μ_{t\|t−1} = C_σ·σ_t² + V^σ_t |
| C | C_μ = 1 − Σφ_i = **0.02730** | C_σ = 1/κ = **0.06581** |
| R | R_t = σ̂²_{t\|t−1}/κ (sale de la creencia previa sobre σ²) | R_σ = Var(ε_t²) en train = **6.49·10⁻¹⁰** |

Aquí κ = γ₀/σ_ε² = **15.19** (train). Es cuántas veces la varianza del spread alrededor de su media excede la varianza del choque diario ε_t en un AR(p).

Las ecuaciones son las mismas del filtro 1:
- **Predecir:** x̂_{t|t−1} = x̂_{t−1|t−1}; P_{t|t−1} = P_{t−1|t−1} + Q.
- **Actualizar:** ỹ_t = y_t − C·x̂_{t|t−1}; S_t = C·P_{t|t−1}·C + R; K_t = P_{t|t−1}·C/S_t; x̂_{t|t} = x̂_{t|t−1} + K_t·ỹ_t; P_{t|t} = (1 − K_t·C)·P_{t|t−1}.
- Se aplican primero a μ y luego a σ², con la innovación ỹ_t de μ como insumo de la medición u_t de σ².

**¿Por qué la observación de μ es y_t y no S_t?**
- El spread está muy autocorrelacionado: φ suma 0.973 en train.
- Si el filtro observara S_t directo, pasaría una de dos cosas. O sus innovaciones heredarían esa autocorrelación, o el filtro tendría que perseguir a S_t con una Q grande, y entonces se comería la reversión que queremos operar.
- Con y_t, la innovación ỹ_t es el **error de pronóstico a un día** del AR(p). Ese sí debe ser ruido blanco, y es lo que mide la prueba obligatoria 4.

**¿Por qué u_t mide σ²?** Se cumple que E[ỹ_t²] = C_μ²·P^μ_{t|t−1} + σ_ε². Restando el primer término queda una medición insesgada de σ_ε² = σ_t²/κ.

**Salida (código: `prior_zscore`, `kalman/filter2_spread.py:109-116`):**

z_t = (S_t − μ̂_{t|t−1}) / σ̂_{t|t−1}

Se usan la media y la varianza **previas**, es decir lo que el filtro creía al cierre de t−1 propagado un paso. La media posterior μ̂_{t|t} ya se movió hacia S_t. Usarla sería juzgar la observación del día con la propia observación del día: el z sale encogido y hay look-ahead. En los datos, el |z| promedio con estadísticos posteriores es menor que con los previos; lo verifica `test_zscore_uses_prior_not_posterior_statistics`.

### 1.3 Calibración: φ, R, Q (solo train, 1,423 días del 2014-05-07 al 2019-12-30)

**Orden p y coeficientes φ (AR por MCO con constante).** La regla parte del BIC y sube p, como en la verificación de Box-Jenkins, hasta que los residuales de train son blancos. "Blanco" significa Ljung-Box(10) p > 0.05 y |acf1| < 1.96/√n.

| p | acf1 de los residuales | Ljung-Box(10) p | ¿Blanco? |
|---|---|---|---|
| 1 (mínimo del BIC: −11693.98) | −0.0655 (banda ±0.052) | 0.043 | no |
| 2 (BIC −11693.88, empate práctico con p = 1) | −0.0003 | 0.043 | no |
| 3 | −0.0024 | 0.034 | no |
| 4 | −0.0008 | 0.034 | no |
| 5 | −0.0038 | 0.037 | no |
| **6** | −0.0052 | **0.118** | **sí → p = 6** |

- φ = (0.894, 0.051, 0.020, −0.053, −0.006, 0.066); Σφ = 0.973.
- σ_ε² = 1.457·10⁻⁵; κ = 15.19.
- Declaración honesta: la primera regla era solo BIC, y daba p = 1. La subí a Box-Jenkins al ver que los residuales del AR(1) tenían autocorrelación de orden 1 fuera de la banda. Está escrita en el código y usa solo train, pero se decidió después de ver ese resultado de train.

**R.**
- R_t de la media = σ̂²_{t|t−1}/κ. El ruido de la observación de μ es el choque ε_t, y su varianza es justo lo que estima el bloque de σ², así que no se fija a mano.
- R_σ = varianza muestral de ε_t² en train (6.49·10⁻¹⁰). Se mide con los datos, igual que el R del filtro 1, y capta colas más pesadas que la normal.

**Q (máxima verosimilitud en rejilla).**
- Q_μ = q_μ·Var(S_train) y Q_σ = q_σ·(σ₀²)².
- Para cada par se maximiza la log-verosimilitud gaussiana de las innovaciones en train. La rejilla es de 34 × 34 = 1,156 pares, con 0 y 33 valores de 10⁻⁸ a 1.
- Aquí la máxima verosimilitud sí es adecuada, a diferencia del filtro 1: el término AR ya explica la reversión, así que una μ que persiga a S_t no mejora el pronóstico a un día y la verosimilitud no la premia.

| | Elegido | ¿Orilla? | Región a ≤ 1.92 de log-verosimilitud (IC 95%) |
|---|---|---|---|
| q_μ | **3.16·10⁻⁴** (Q_μ = 7.13·10⁻⁸) | no | **[0, 10⁻²]: plana, débilmente identificada** |
| q_σ | **5.62·10⁻³** (Q_σ = 2.76·10⁻¹⁰) | no | [3.2·10⁻³, 10⁻²] |

**Sensibilidad a q_μ (train, sin burn-in, q_σ fijo).** La verosimilitud apenas distingue entre valores de q_μ, pero la señal cambia mucho:

| q_μ | Log-verosimilitud | Desviación estándar de z | Días con \|z\| > 2 |
|---|---|---|---|
| 0 | 5901.59 | 1.026 | 45 |
| 10⁻⁴ | 5901.66 | 0.988 | 39 |
| **3.16·10⁻⁴ (elegido)** | **5901.70** | **0.929** | **18** |
| 10⁻³ | 5901.60 | 0.814 | 8 |
| 10⁻² | 5899.84 | 0.406 | 0 |

- Me quedo con la regla escrita (máxima verosimilitud) y no elijo q_μ por número de operaciones: eso sería calibrar la estrategia, y le toca al grid search de Paula en train.
- Lo marco como pregunta para el profesor.

**Inicialización (creencias de train):**
- μ₀ = 0, porque los log-precios del filtro 1 están centrados en su media de train.
- P^μ₀ = Var(S_train) = 2.25·10⁻⁴.
- σ₀² = κ·σ_ε² = 2.21·10⁻⁴.
- P^σ₀ = (σ₀²)², es decir, la desviación estándar de la creencia es el 100% del valor.

**Burn-in.** Termina cuando se cumplen dos condiciones, y manda la más tardía:
- **Media:** P^μ_{t|t} se queda dentro de su banda estable de train, con el mismo criterio que el filtro 1 (`measure_burn_in`). Se cumple a los 262 días.
- **Varianza:** P^σ_{t|t} se queda a menos de 1% de su equilibrio de Riccati, 6.29·10⁻⁹. Se cumple a los 65 días.
- **¿Por qué la varianza tiene otro criterio?** P^σ no depende de los datos, porque C_σ, Q_σ y R_σ son constantes, y converge a un valor fijo. Con la banda mediana ± 3·MAD su MAD es casi cero, y el día de corte cambiaba con el redondeo de punto flotante: daba 2015-12-16 con los artefactos de Milca y 2016-01-05 con los regenerados en otra máquina. Con el criterio de Riccati el resultado es idéntico en ambos casos.

Resultado:
- El filtro arranca el 2014-05-07, al terminar el burn-in del filtro 1. Necesita 6 rezagos, así que el primer z es del 2014-05-15.
- El burn-in dura **262 días** y termina el **2015-05-21**.
- Se puede operar en train del 2015-05-21 al 2019-12-30: 1,161 días, el 51% de train.
- Respaldos o fallbacks: 0.

### 1.4 Ganancia de Kalman: cálculo e interpretación

K_t = P_{t|t−1}·C / (C·P_{t|t−1}·C + R) indica qué fracción de la sorpresa del día se cree:
- **Media μ (K_μ):** empieza en 0.42 y, después del burn-in, oscila entre 0.002 y 0.28, con promedio 0.058. Sube cuando la varianza estimada baja, porque R_t es menor y se confía más en el dato, y cae en marzo de 2020, cuando σ̂² se dispara. La media se mueve como máximo C_μ·K_μ ≈ 0.15% de cada sorpresa: μ es casi un ancla.
- **Varianza σ² (K_σ):** C_σ y R_σ son constantes, así que la ganancia converge a un estado estacionario: K_σ = 0.638, o sea C_σ·K_σ = 4.2% de cada medición nueva. Equivale a una memoria de unos 24 días hábiles. Por eso en la figura 5 es una línea plana después de unas semanas.
- La figura 5 grafica |K_t| de los 5 pesos del filtro 1 y de μ y σ² del filtro 2, en escala **lineal**.

### 1.5 Ejemplo numérico de 5 periodos (train, primeros días del filtro 2)

Parámetros: C_μ = 0.0273041, C_σ = 0.0658141, κ = 15.1943, Q_μ = 7.1295·10⁻⁸, Q_σ = 2.7561·10⁻¹⁰, R_σ = 6.4887·10⁻¹⁰.

| Fecha | 2014-05-15 | 2014-05-16 | 2014-05-19 | 2014-05-20 | 2014-05-21 |
|---|---|---|---|---|---|
| S_t | 0.0129729 | 0.0075861 | 0.0090204 | 0.0113120 | 0.0145364 |
| μ̂_{t\|t−1} | 0 | −0.0021900 | −0.0037410 | −0.0032036 | −0.0021776 |
| P^μ_{t\|t−1} | 2.25526·10⁻⁴ | 2.23025·10⁻⁴ | 2.21023·10⁻⁴ | 2.19122·10⁻⁴ | 2.16930·10⁻⁴ |
| σ̂²_{t\|t−1} | 2.21384·10⁻⁴ | 2.69293·10⁻⁴ | 2.78007·10⁻⁴ | 2.37860·10⁻⁴ | 2.19226·10⁻⁴ |
| **z_t (previos)** | **0.8719** | **0.5957** | **0.7654** | **0.9412** | **1.1289** |
| y_t | −0.0052417 | −0.0046164 | 0.0015421 | 0.0026250 | 0.0034101 |
| ỹ_t (μ) | −0.0052417 | −0.0045566 | 0.0016443 | 0.0027124 | 0.0034696 |
| S_t (var. de innovación) | 1.47383·10⁻⁵ | 1.78895·10⁻⁵ | 1.84615·10⁻⁵ | 1.58179·10⁻⁵ | 1.45899·10⁻⁵ |
| K_μ | 0.41781 | 0.34039 | 0.32689 | 0.37824 | 0.40597 |
| μ̂_{t\|t} | −0.0021900 | −0.0037410 | −0.0032036 | −0.0021776 | −0.0007691 |
| P^μ_{t\|t} | 2.22954·10⁻⁴ | 2.20952·10⁻⁴ | 2.19051·10⁻⁴ | 2.16859·10⁻⁴ | 2.14526·10⁻⁴ |
| u_t | 2.73068·10⁻⁵ | 2.05962·10⁻⁵ | 2.5388·10⁻⁶ | 7.19387·10⁻⁶ | 1.18761·10⁻⁵ |
| K_σ | 3.76151 | 3.03303 | 2.54773 | 2.20235 | 1.94483 |
| σ̂²_{t\|t} | 2.69293·10⁻⁴ | 2.78007·10⁻⁴ | 2.37860·10⁻⁴ | 2.19226·10⁻⁴ | 2.14263·10⁻⁴ |
| P^σ_{t\|t} | 3.70851·10⁻⁸ | 2.99029·10⁻⁸ | 2.51183·10⁻⁸ | 2.17131·10⁻⁸ | 1.91743·10⁻⁸ |

**Primer periodo, recalculado a mano:**
1. **Predicción.** P^μ_{t|t−1} = 2.25455·10⁻⁴ + 7.13·10⁻⁸ = 2.25526·10⁻⁴.
2. **z con estadísticos previos.** z = (0.0129729 − 0)/√2.21384·10⁻⁴ = 0.0129729/0.0148790 = 0.8719.
3. **Innovación.** ỹ = −0.0052417 − 0.0273041·0 = −0.0052417. R_t = 2.21384·10⁻⁴/15.1943 = 1.45702·10⁻⁵.
4. **Varianza de la innovación.** S = 0.0273041²·2.25526·10⁻⁴ + 1.45702·10⁻⁵ = 1.68131·10⁻⁷ + 1.45702·10⁻⁵ = 1.47383·10⁻⁵.
5. **Ganancia.** K_μ = 2.25526·10⁻⁴·0.0273041/1.47383·10⁻⁵ = 0.41781.
6. **Actualización de μ.** μ̂ = 0.41781·(−0.0052417) = −0.0021900; P^μ = (1 − 0.41781·0.0273041)·2.25526·10⁻⁴ = 2.22954·10⁻⁴.
7. **Medición de la varianza.** u = 0.0052417² − 1.68131·10⁻⁷ = 2.73068·10⁻⁵. Su sorpresa es u − C_σ·σ̂² = 2.73068·10⁻⁵ − 1.45702·10⁻⁵ = 1.27367·10⁻⁵.
8. **Ganancia de σ².** S_σ = 0.0658141²·4.92864·10⁻⁸ + 6.4887·10⁻¹⁰ = 8.6235·10⁻¹⁰, así que K_σ = 4.92864·10⁻⁸·0.0658141/8.6235·10⁻¹⁰ = 3.7615.
9. **Actualización de σ².** σ̂² = 2.21384·10⁻⁴ + 3.7615·1.27367·10⁻⁵ = 2.69293·10⁻⁴.

**Lectura de la tabla:**
- El primer día el spread queda debajo de lo que predice el AR: ỹ < 0. Por eso μ̂ baja.
- La sorpresa al cuadrado (2.7·10⁻⁵) es mayor que la esperada (1.46·10⁻⁵), así que σ̂² sube.
- K_σ va bajando (3.76 → 1.94) conforme el filtro gana confianza: P^σ baja.
- La prueba `test_first_five_real_periods_by_hand_match_artifacts` repite estos 5 periodos con aritmética escalar, sin llamar al filtro, y coincide con los artefactos con tolerancia relativa de 10⁻⁹.

### 1.6 Innovaciones (prueba obligatoria 4)

El criterio está escrito en `tests/test_innovations.py`:
- varianza de ỹ_t/√S_t en [0.8, 1.2];
- Ljung-Box(10) con p > 0.05;
- |acf1| < 1.96/√n.

| Tramo (sin burn-in) | n | Varianza | acf1 | Ljung-Box(10) p | ¿Cumple? |
|---|---|---|---|---|---|
| **train** | 1,161 | **1.061** | **0.003** (banda ±0.058) | **0.450** | **sí (es la prueba)** |
| test | 754 | 1.091 | 0.106 | 0.140 | varianza y Ljung-Box sí, acf1 no |
| valid | 755 | 1.394 | 0.063 | 0.027 | no |

- En test y valid las innovaciones se deterioran: la varianza sube a 1.39 en valid y Ljung-Box rechaza al 5%.
- No se recalibra con esos tramos, porque estaría prohibido. Se reporta como evidencia de cambio de régimen, coherente con el aviso de Milca de que las últimas 3 sub-ventanas de train ya no eran estables.
- **Filtro 1 (diagnóstico, no prueba estricta):** sus innovaciones normalizadas tienen acf1 = **0.967** y varianza 0.795 (train, sin burn-in, 1,423 días). No es un error. La innovación del filtro 1 **es** el spread (S_t = ỹ_t), y el spread revierte con vida media de 15 a 20 días. Si fueran blancas, el filtro 1 habría perseguido cada desviación y no quedaría reversión que operar. La prueba de ruido blanco aplica al filtro 2. **Punto a confirmar con el profesor.**

### 1.7 Formulación de Powell del filtro 2

**Proceso de 6 pasos:**
1. **Definir el problema:** saber cada día, sin ver el futuro, dónde está el equilibrio del spread y qué tanto suele alejarse, para medir la desviación en unidades comparables (z).
2. **Modelar el estado:** la creencia B_t = (μ̂_{t|t}, P^μ_{t|t}, σ̂²_{t|t}, P^σ_{t|t}) sobre la media y la varianza latentes.
3. **Modelar la transición:** caminatas aleatorias lentas, μ_t = μ_{t−1} + W^μ y σ²_t = σ²_{t−1} + W^σ, con Q_μ y Q_σ por máxima verosimilitud en train.
4. **Modelar la observación:** y_t = C_μ·μ_t + ε_t con un AR(6), y u_t = C_σ·σ_t² + V^σ_t.
5. **Definir la regla de decisión:** publicar z_t = (S_t − μ̂_{t|t−1})/σ̂_{t|t−1}, con estadísticos previos.
6. **Definir el mecanismo de aprendizaje:** las ecuaciones de actualización de Kalman, con ganancias K_μ y K_σ.

**Cinco elementos:**

| Elemento | Filtro 2 |
|---|---|
| Variables de estado | Creencia (μ̂, P^μ, σ̂², P^σ); información: S_{t−1}, ..., S_{t−6} y los parámetros congelados de train (φ, κ, R_σ, Q) |
| Variables de decisión | El z_t publicado, o sea qué estadísticos usar: los previos |
| Información exógena | El spread nuevo S_{t+1}, que llega del filtro 1 con los precios de t+1 |
| Función de transición | Predicción (sumar Q) y actualización de Kalman de μ y luego de σ² |
| Función objetivo | Maximizar la log-verosimilitud Σ_t −½[ln(2π·S_t) + ỹ_t²/S_t], equivalente a pronósticos a un día insesgados con varianza bien calibrada |

## 2. VECM (`strategy/vecm.py`)

### 2.1 Del VAR(p) al VECM

- **VAR(p):** P_t = c + A_1·P_{t−1} + ... + A_p·P_{t−p} + ε_t.
- Se resta P_{t−1} de ambos lados y se reacomodan los rezagos: ΔP_t = c + Π·P_{t−1} + Σ_{i=1}^{p−1} Γ_i·ΔP_{t−i} + ε_t, con Π = Σ_i A_i − I y Γ_i = −Σ_{j>i} A_j.
- **Con r relaciones de cointegración**, Π tiene rango r y se escribe Π = α·βᵀ:
  - βᵀP_{t−1} es el desequilibrio de ayer, el término de corrección de error (ECT);
  - α dice cuánto corrige cada activo por día hacia el equilibrio.

### 2.2 Especificación y estimación (train, 2,262 observaciones)

- Usé la misma especificación que Milca: **p = 1** (k_ar_diff = 0 rezagos de ΔP), **r = 1** y **constante sin restringir** (caso 3).
- La verifiqué con `select_order` de statsmodels en train: AIC, BIC, HQIC y FPE eligen todos k_ar_diff = 0. Es consistente con p = 1.
- **β se fija en el vector de Johansen de train** (XEL +1, ES −0.658, ED −0.484, AEE −0.321, DUK +0.186, CMS +0.131). α y c se estiman por MCO ecuación por ecuación (statsmodels OLS). Dado β, eso es el estimador de máxima verosimilitud del VECM.

| Activo | β (Johansen) | α | t de α |
|---|---|---|---|
| DUK | +0.186 | +0.0146 | 1.32 |
| XEL | +1 | +0.0024 | 0.21 |
| ED | −0.484 | +0.0371 | **3.29** |
| ES | −0.658 | +0.0391 | **3.20** |
| CMS | +0.131 | +0.0155 | 1.27 |
| AEE | −0.321 | +0.0243 | 1.90 |

**Lectura:**
- βᵀα = **−0.0443**, negativo: el desequilibrio se corrige. La vida media implícita es −ln 2/ln(1 − 0.0443) = **15.3 días** (train), igual a la vida media de Johansen que reportó Milca.
- La corrección la hacen sobre todo ED y ES. Tienen β < 0 y α > 0 significativo: cuando el ECT está arriba, suben, y como pesan con signo negativo, bajan el ECT.
- XEL casi no corrige (t = 0.21): es débilmente exógena, y las demás patas se ajustan a ella.
- Control: la clase VECM de statsmodels, sin fijar β, estima un vector que difiere del de Johansen como máximo en 0.055 (DUK: 0.130 contra 0.186). Se debe al tratamiento distinto de la constante. Los signos coinciden.
- Artefacto: `artifacts/strategy/vecm_summary.json`, que incluye Γ (vacío con p = 1), Σ_u y R².

## 3. Z-score de señales (`strategy/zscore.py`)

El enunciado pide un z-score que "se deriva de los residuales del VECM, normalizados con las estimaciones en tiempo real del filtro 2". La frase admite tres lecturas:

| Opción | Qué se normaliza | Evidencia (train) | Veredicto |
|---|---|---|---|
| **A (elegida)** | El residual de **equilibrio** del VECM con los pesos dinámicos: ECT_t = β̂_tᵀ(log P_t − μ_train) | Es lo que el VECM corrige; revierte (acf1 del spread ≈ 0.97) | Defendible |
| B | Las innovaciones ε_t de cada ecuación, proyectadas con β | acf1 = **0.008**: ruido blanco, no revierte | No hay nada que operar |
| C | El ECT con el β estático de Johansen | Correlación con A = **0.60** (train, sin burn-in) | Ignora el filtro 1, que el enunciado pide usar |

**Supuesto tomado:** la opción A. Es el término de corrección de error del VECM evaluado con el β̂_t del filtro 1, con los pesos previos y sin look-ahead, y coincide con el spread que observa el filtro 2: la diferencia máxima es 5.6·10⁻¹⁶. Por eso el z de señales es exactamente z_t de `z_t.csv`; lo verifica `test_signal_zscore_from_vecm_residual_matches_filter2_zscore`. **Pregunta para el profesor:** ¿es esta la lectura que espera?

## 4. Lógica de señales (`strategy/signals.py`)

### 4.1 Regla (diapositiva del profesor) como máquina de estados

h_t es el número de días que lleva abierta la posición.

| Condición | d_t |
|---|---|
| d_{t−1} = 0 y z_t < −z_in | +1 (largo el spread) |
| d_{t−1} = 0 y z_t > +z_in | −1 (corto el spread) |
| \|z_t\| < z_out, o \|z_t\| > z_stop, o h_t ≥ h_max | 0 (cerrar) |
| cualquier otro caso | d_{t−1} (mantener) |

- Solo se entra desde plano: para voltear, primero se cierra.
- Si en plano |z_t| > z_stop, las condiciones de entrar y de cerrar se cumplen a la vez. Manda la de cerrar: no se abre una posición que el stop cerraría ese mismo día.
- Cuando coinciden varias salidas, el motivo se asigna en este orden: stop, luego señal, luego tenencia máxima.
- d_t se decide con el cierre de t y se aplica del cierre de t al de t+1, en el backtest.
- **Firma:** `generate_positions(z, params)` → DataFrame con `position` (d_t ∈ {−1, 0, +1}), `holding_days` (h_t) y `exit_reason` (`senal`, `stop`, `tenencia_maxima` o vacío).
- **Dos comportamientos literales de la regla que conviene conocer:**
  - Si z salta de +2.5 a −2.5 sin pasar por |z| < z_out, la posición se mantiene, porque no se cumple ninguna condición de cierre.
  - Después de un cierre por tenencia máxima, si z sigue más allá de z_in, al día siguiente se vuelve a entrar.
  - Ambos casos están cubiertos en `tests/test_signals.py`.

### 4.2 Parámetros por defecto (`artifacts/strategy/params_default.json`)

Son valores de arranque convencionales, no óptimos: **z_entry = 2.0, z_exit = 0.5, z_stop = 3.0, max_hold_days = 31**.
- **Cálculo de la tenencia máxima a partir de la vida media:** con reversión exponencial, el desequilibrio se reduce a la mitad cada vida media. Pasar de z_entry a z_exit toma vida_media·log₂(z_entry/z_exit) = 15.3·log₂(4) = 30.6 días, que se redondea a **31 días hábiles**. Si para entonces no revirtió, la posición dejó de revertir.
- **Sensibilidad de la tenencia máxima:** con la vida media del spread de Kalman (20.4 días, train) saldrían 41 días.
- **Número de operaciones con los valores por defecto:**

| z_entry | train | test | valid |
|---|---|---|---|
| 1.5 | 12 | 8 | 10 |
| **2.0** | **6** | 3 | 3 |
| 2.5 | 1 | 2 | 1 |

  Las columnas test y valid son solo descriptivas; no se usaron para elegir nada. Con 6 operaciones en train la muestra es chica, por eso el grid search de Paula debe exigir un número mínimo de operaciones.
- **Los definitivos los elige Paula** con grid search **solo en train**, con restricción de número mínimo de operaciones, usando `generate_positions`.

### 4.3 Verificación de la dirección (train)

**Regla de dirección:** largo el spread = comprar cada activo con β̂_{t,i} > 0 y vender cada uno con β̂_{t,i} < 0, en proporción a |β̂_{t,i}|. Corto el spread = todos los signos al revés.

**Episodio conocido:** corto en el spread del 2019-08-28 al 2019-09-09, en train.
- **Entrada:** z = 2.04, con el spread en 0.0406 contra una media previa de 0.0084. XEL está cara frente a la canasta.
- **Pesos del día:** XEL +1, DUK +0.066, ED −0.446, ES −0.313, CMS −0.160 y AEE −0.285. Corto en el spread significa **vender XEL y DUK y comprar ED, ES, CMS y AEE**.
- **Rendimientos de las patas durante la operación:**
  - XEL −0.93% (vendida: gana);
  - ED +2.94% y ES +1.86% (compradas: ganan);
  - DUK +1.53% (vendida: pierde);
  - CMS −1.65% y AEE −0.76% (compradas: pierden).
- **Salida por señal** el 2019-09-09, con |z| = 0.43 < 0.5 y el spread en 0.0145.
- **Resultado bruto:** +0.99% por unidad de nocional bruto. Con todos los signos al revés habría sido −0.99%.
- **CMS:** su peso es **negativo** en el filtro (≈ −0.29 en train después del burn-in) aunque en Johansen es +0.131. Aquí se usó el β̂_t del día, así que CMS se compra en un corto del spread.

**Agregado (train, sin costos ni sizing; no es desempeño, solo el signo):**
- corr(z_t, rendimiento del spread al día siguiente) = **−0.077**. Negativa, como debe ser: z alto anticipa una baja del spread.
- **5 de 6** operaciones con rendimiento bruto positivo: 5 cortas y 1 larga. La larga, del 2015-09-18 al 2015-10-14, rindió +1.06%.
- Suma de rendimientos: +3.06% con la regla y −3.06% con los signos al revés. Artefacto: `artifacts/strategy/direction_check.json`.
- El episodio que sugirió Milca (2014-05-07 a 2014-05-13, spread positivo y creciente) cae en el burn-in del filtro 2. No genera señal, pero su z es positivo (0.87 el 2014-05-15), con el signo correcto.

## 5. Formulación de Powell del sistema completo (filtros 1 y 2, VECM y trading)

**Proceso de 6 pasos:**
1. **Definir el problema:** decidir cada día si estar largo, corto o plano en el spread de las 6 eléctricas, para ganar con la reversión al equilibrio después de costos, sin usar información futura.
2. **Modelar el estado:** S_t = (R_t, I_t, B_t), con las tres componentes de la tabla de abajo.
3. **Modelar la transición:** pesos y media latentes como caminatas aleatorias (Q₁, Q_μ, Q_σ) y el desequilibrio con corrección de error (VECM: ΔP_t = c + α·βᵀP_{t−1} + ε_t). La posición evoluciona según la máquina de estados.
4. **Modelar la observación:** precios P_{t+1}. El filtro 1 los convierte en S_{t+1}, y el filtro 2 observa y_{t+1} y u_{t+1}.
5. **Definir la regla de decisión:** política de umbrales X^π(S_t | θ), con θ = (z_in, z_out, z_stop, h_max). Es una *policy function approximation*.
6. **Definir el mecanismo de aprendizaje:** dentro de cada día, las actualizaciones de Kalman de ambos filtros; entre corridas, la calibración de θ por grid search en train (Paula).

**Estado S_t = (R_t, I_t, B_t):**

| Componente | Contenido |
|---|---|
| **Física R_t** | Posición d_{t−1} ∈ {−1, 0, +1}, días abierta h_t, unidades en cada pata, efectivo y préstamo acumulado sobre las patas cortas (contabilidad del backtest) |
| **Información I_t** | Log-precios P_t y rezagos S_{t−1}, ..., S_{t−6}; constantes de train: μ_train, β y α del VECM, φ, κ, R, R_σ, Q₁, Q_μ, Q_σ y θ |
| **Creencia B_t** | Filtro 1: (ẑ_{t\|t}, P_{t\|t}), es decir β̂ y su incertidumbre. Filtro 2: (μ̂_{t\|t}, P^μ_{t\|t}, σ̂²_{t\|t}, P^σ_{t\|t}) |

**Cinco elementos:**

| Elemento | Sistema completo |
|---|---|
| Variables de estado | S_t = (R_t, I_t, B_t) de la tabla |
| Variables de decisión | x_t = d_t ∈ {−1, 0, +1}; las patas se dimensionan ∝ β̂_t con el 80% del capital (sizing de Paula) |
| Información exógena | W_{t+1} = P_{t+1}, los precios del día siguiente |
| Función de transición | B_{t+1} = Kalman₁ y luego Kalman₂ (con S_{t+1}); R_{t+1} = máquina de estados, valuación de patas, comisiones y devengo; I_{t+1} = rezagos desplazados |
| Función objetivo | max_θ E[Σ_t (PnL_{t+1} − comisiones − préstamo)], en la práctica el Sharpe neto en train con número mínimo de operaciones |

## 6. Figuras (`artifacts/figures/`)

Todas tienen título, ejes etiquetados, leyenda, fronteras train/test/valid dibujadas y etiquetadas, y el burn-in sombreado.
- **Figura 2** (`fig2_spread_senales.png`): spread S_t con la media previa μ̂_{t|t−1} y la banda μ̂ ± z_entry·σ̂, entradas largas (▲), entradas cortas (▼) y salidas por señal, stop o tenencia máxima, con los parámetros por defecto.
- **Figura 5** (`fig5_ganancia_kalman.png`): |K_t| de los 5 pesos del filtro 1 y de μ y σ² del filtro 2, en escala **lineal**. El máximo del burn-in se anota aparte para no aplastar el resto.
- **Figura 6** (`fig6_zscore.png`): z_t con ±z_entry, ±z_exit y ±z_stop.

## 7. Limitaciones

1. **Las innovaciones del filtro 1 están autocorrelacionadas (0.967, train):** es la reversión del spread, no un defecto. La prueba de blancura aplica al filtro 2. Hay que confirmarlo con el profesor.
2. **Posible ruptura de la cointegración (heredado de Milca):** las 3 sub-ventanas inestables son las últimas de train. Hay señales fuera de muestra:
   - las innovaciones del filtro 2 empeoran en valid (varianza 1.39, Ljung-Box p = 0.027);
   - en marzo de 2024 (valid) el z llega a −3.9;
   - en 2020 (test) σ̂ se triplica (la σ̂ promedio pasa de 0.014 en train a 0.030 en test). La banda se abre y casi no hay señales durante el choque del COVID.
3. **q_μ está débilmente identificado:** la verosimilitud es plana en [0, 10⁻²], y la desviación estándar de z va de 1.02 a 0.40 dentro de esa región. Se respetó la regla de máxima verosimilitud.
4. **El orden del AR (p = 6) salió de una regla Box-Jenkins** que se escribió después de ver que el AR(1) del BIC dejaba autocorrelación de orden 1 (−0.066) en train. Se declara arriba.
5. **El burn-in total es largo:** se puede operar en train solo desde el 2015-05-21 (1,161 de 2,263 días). Con los valores por defecto hay apenas 6 operaciones en train: 5 cortas y 1 larga.
6. **Los parámetros por defecto no están optimizados.** El z con estadísticos previos tiene desviación estándar 0.93 en train, así que z_entry = 2 equivale a unas 2.15 desviaciones observadas.
7. **Pruebas múltiples:** el grupo salió de 910 pruebas de Johansen (Milca). El desempeño final debe corregirse con el Deflated Sharpe Ratio o el Reality Check, en el backtest.
8. **El stop sobre |z| es débil ante un choque, porque σ̂ se adapta (valid).** El 2024-02-29 el spread cae de −0.020 a −0.112, con z = −3.90: no se entra porque está más allá del stop. Al día siguiente el spread está en −0.175, pero σ̂ ya creció y z = −2.18, así que se abre largo. El spread llega a −0.220 y z nunca baja de −2.18. El stop no se dispara y la posición cierra por tenencia máxima el 2024-04-16. La defensa efectiva contra "una posición que deja de revertir" es la tenencia máxima.
9. **Reproducibilidad entre máquinas:** recalcular el filtro 1 en otra computadora cambia el spread en menos de 1.5·10⁻⁹ por redondeo de punto flotante. Dentro de una misma máquina el resultado es idéntico bit a bit.

## 8. Preguntas para el profesor

1. ¿El "residual del VECM" que se normaliza es el residual de equilibrio βᵀP_t (opción A) o las innovaciones ε_t?
2. ¿La prueba de innovaciones blancas aplica solo al filtro 2? Las del filtro 1 son el spread y deben revertir.
3. Después de una salida por tenencia máxima, ¿se permite reentrar al día siguiente si z sigue fuera de la banda, como dice la regla literal?
4. Con la verosimilitud plana en q_μ, ¿es aceptable dejar q_μ en el máximo, o prefiere fijarlo por un criterio de señal en train?
5. ¿Es aceptable subir el orden del AR por diagnóstico de residuales (Box-Jenkins) por encima del mínimo del BIC?
