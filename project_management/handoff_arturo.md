# Handoff · Arturo · filtro de Kalman 2, VECM, z-score y señales

Rama: `arturo/kalman2-strategy`, creada desde `milca/cointegracion-kalman1` porque el PR de Milca todavía no estaba en `main`. Es para Paula: sizing, backtest, métricas y dashboard.

## Cómo se corre

```
python main.py        # ... -> kalman.filter2_spread -> strategy; se detiene en backtest (NotImplementedError de Paula)
python -m pytest      # o: pytest  (pytest.ini en la raíz)
```

- Mis dos etapas tardan unos 20 segundos. La etapa `cointegration` de Milca tardó 40 minutos en mi máquina de 8 núcleos; ella reporta 6.5 minutos en la suya.
- **En Windows, `python main.py` falla en la etapa `data` de Milca** con `UnicodeDecodeError`. Sus archivos leen JSON con `read_text()` sin `encoding="utf-8"`:
  - `cointegration/universe.py:18`;
  - `cointegration/__init__.py:64-65`;
  - `kalman/filter1_weights.py:258-259`.
- No toqué sus archivos. Yo corrí con `PYTHONUTF8=1`. En macOS y Linux no pasa. **Milca debe arreglarlo antes de la entrega**: el profe penaliza con −10 si el repositorio no corre con el comando documentado.

## Lo que Paula necesita

1. **Columna del z-score:** `z` de `artifacts/kalman/z_t.csv`. Solo se opera donde `burn_in == False`, desde el 2015-05-21. Antes de esa fecha el z es NaN o está en burn-in.
2. **Firma:**
   ```python
   from strategy import generate_positions
   positions = generate_positions(z, params)   # z: pd.Series por fecha; params: dict
   ```
   - `params` lleva `z_entry`, `z_exit`, `z_stop` y `max_hold_days`, en días hábiles, es decir filas.
   - Devuelve un DataFrame con el mismo índice y estas columnas:
     - `position`: d_t ∈ {−1, 0, +1};
     - `holding_days`: h_t;
     - `exit_reason`: `senal`, `stop`, `tenencia_maxima` o vacío.
   - Pásale `z.where(~burn_in)`: con z NaN nunca abre posición.
3. **Temporalidad:** d_t se decide con el cierre de t y se aplica del cierre de t al de t+1. No apliques d_t al rendimiento de t, porque eso es look-ahead.
4. **Parámetros por defecto** (`artifacts/strategy/params_default.json`): z_entry = 2.0, z_exit = 0.5, z_stop = 3.0 y max_hold_days = 31.
   - 31 = ⌈15.3·log₂(2/0.5)⌉, con la vida media de Johansen de train.
   - **Son de arranque, no óptimos.** Los definitivos salen de tu grid search **solo en train**, con un número mínimo de operaciones.
   - Con los valores por defecto hay 6 operaciones en train, 3 en test y 3 en valid. Con z_entry = 1.5 serían 12, 8 y 10.
5. **Regla de dirección completa:** largo el spread (d = +1) = comprar cada activo con β̂_{t,i} > 0 y vender cada uno con β̂_{t,i} < 0. Corto el spread (d = −1) = todos los signos al revés.
   - El nocional de cada pata es proporcional a |β̂_{t,i}|. Son log-precios, así que el peso es exposición en valor, no número de acciones.
   - Usa **β̂_t de `artifacts/kalman/beta_t.csv` del día**. No uses el β estático de Johansen.
6. **Advertencia sobre CMS:** en Johansen vale +0.131, pero en el filtro vale ≈ −0.29 en train después del burn-in, sin cambios de signo. Con el β estático la dirección de CMS sale al revés.
7. **Verificación de signo hecha** (`artifacts/strategy/direction_check.json`, train, sin costos):
   - corr(z, rendimiento del spread al día siguiente) = −0.077;
   - 5 de 6 operaciones con rendimiento bruto positivo;
   - +3.06% con la regla y −3.06% con los signos al revés.
   - Si tu backtest pierde de forma sostenida con estos parámetros, revisa el signo antes que nada.
8. `artifacts/strategy/positions_default.csv` trae las posiciones con los valores por defecto, para la figura 2 y para comparar. En el CSV, `exit_reason` vacío se lee como NaN.

## Qué hice y por qué

- **Filtro 2:**
  - El estado es la media latente μ_t y la varianza latente σ_t².
  - La media se observa a través de un AR(6) del spread: y_t = S_t − Σφ_i·S_{t−i}. Así la innovación es un pronóstico a un día y queda blanca.
  - La varianza se observa con u_t = ỹ_t² − C_μ²·P^μ.
  - z_t = (S_t − μ̂_{t|t−1})/σ̂_{t|t−1}, con estadísticos **previos** (`kalman/filter2_spread.py:109-116`).
- **Calibración (solo train, del 2014-05-07 al 2019-12-30):**
  - p por BIC y luego Box-Jenkins: p = 6.
  - R_σ = Var(ε²).
  - (q_μ, q_σ) por máxima verosimilitud en una rejilla de 1,156 pares: q_μ = 3.16·10⁻⁴ y q_σ = 5.62·10⁻³, ninguno en la orilla.
  - Burn-in: 262 días, hasta el 2015-05-21.
- **VECM:**
  - Especificación: p = 1, r = 1 y constante sin restringir. `select_order` lo confirma con todos los criterios.
  - β se fija en el de Johansen y α se estima por MCO.
  - βᵀα = −0.0443, que da una vida media de 15.3 días (train).
  - ED y ES son las patas que corrigen (t = 3.3 y 3.2). XEL casi no corrige (t = 0.2).
- **Z-score de señales:** es el residual de equilibrio del VECM, ECT_t = β̂_tᵀ(P_t − μ_train), normalizado con el filtro 2. Es idéntico al z de `z_t.csv`. Las alternativas y la justificación están en `docs/secciones/arturo.md` §3.
- **Señales:** la regla literal de la diapositiva, como máquina de estados. Entradas solo desde plano; no se entra si |z| > z_stop.

## Cifras clave

| Cifra | Valor | Tramo |
|---|---|---|
| Innovaciones normalizadas del filtro 2: varianza / acf1 / Ljung-Box(10) p | 1.061 / 0.003 / 0.450 | train (1,161 días) |
| Lo mismo | 1.091 / 0.106 / 0.140 | test |
| Lo mismo | 1.394 / 0.063 / 0.027 | valid |
| acf1 de las innovaciones del filtro 1 | 0.967 | train |
| Desviación estándar de z | 0.929 / 0.953 / 0.923 | train / test / valid |
| σ̂ promedio | 0.0143 / 0.0300 / 0.0329 | train / test / valid |
| Vida media implícita del VECM | 15.3 días | train |

## Archivos que produzco

| Archivo | Contenido |
|---|---|
| `artifacts/kalman/z_t.csv` | `z`, `mu_prev`, `sigma_prev`, `spread`, `innovation_normalized`, `period`, `burn_in` |
| `artifacts/kalman/gain_2.csv`, `posvar_2.csv` | K_t y P_{t\|t} de μ (`mu`) y de σ² (`sigma2`) + `period`, `burn_in` |
| `artifacts/kalman/filter2_params.json`, `filter2_q_grid.csv` | Parámetros, diagnósticos por tramo y rejilla de verosimilitud |
| `artifacts/strategy/params_default.json` | z_entry, z_exit, z_stop, max_hold_days y la regla de la tenencia máxima |
| `artifacts/strategy/vecm_summary.json` | α, t de α, c, Σ_u, R², select_order, β libre contra β fijo y las opciones de residual |
| `artifacts/strategy/signal_zscore.csv` | ECT_t y su z |
| `artifacts/strategy/positions_default.csv`, `direction_check.json` | Posiciones con los valores por defecto y la verificación de signo |
| `artifacts/figures/fig2_spread_senales.png`, `fig5_ganancia_kalman.png`, `fig6_zscore.png` | Figuras 2, 5 y 6 |

Todas las salidas con fecha tienen la fecha ISO en la primera columna, columnas en minúsculas y `period`.

## Supuestos

- El "residual del VECM" es el residual de equilibrio βᵀP_t, evaluado con β̂_t (opción A). Es una pregunta para el profesor.
- La prueba de innovaciones blancas aplica al filtro 2. Las del filtro 1 son el spread.
- Train es "en muestra" para el filtro 2. Test y valid se corren con parámetros congelados.

## Límites conocidos

1. q_μ está débilmente identificado: la verosimilitud es plana en [0, 10⁻²] y la desviación estándar de z cambia de 1.03 a 0.41 en esa región.
2. El orden p = 6 viene de una regla Box-Jenkins escrita después de ver que el AR(1) dejaba acf1 = −0.066 en train.
3. Fuera de muestra las innovaciones se deterioran (valid: varianza 1.39, Ljung-Box p = 0.027). Junto con las 3 sub-ventanas inestables de Milca al final de train, es un aviso de ruptura.
4. Pocas operaciones con los valores por defecto (6 en train).
5. La regla literal mantiene la posición si z cruza de un lado al otro sin pasar por |z| < z_exit, y reentra al día siguiente después de una salida por tenencia máxima.
6. **El stop sobre |z| es débil ante un choque, porque σ̂ se adapta.** Ejemplo en valid:
   - El 2024-02-29 el spread cae de −0.020 a −0.112, con z = −3.90. Está más allá del stop, así que no se entra.
   - Al día siguiente el spread está en −0.175, todavía peor, pero σ̂ ya creció y z = −2.18: se abre largo.
   - Durante la operación z nunca baja de −2.18, aunque el spread llega a −0.220. El stop no se dispara y la posición cierra por tenencia máxima el 2024-04-16.
   - La defensa efectiva contra una posición que deja de revertir es max_hold_days. Paula: tómalo en cuenta en la rejilla (por ejemplo, el rango de max_hold_days o un stop sobre el nivel del spread, si el equipo lo acuerda).
