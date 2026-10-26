# Contrato entre módulos

Archivos que se pasan los módulos. Si necesitas un cambio en un archivo de otra persona, déjalo como comentario o issue; no lo edites.

## Reglas de formato

- Fechas en ISO `YYYY-MM-DD` en la primera columna.
- Nombres de columnas en minúsculas con guion bajo.
- Una columna que etiquete el periodo como `train`, `test` o `valid`.

## Archivos

| De → a | Archivo | Contenido |
|---|---|---|
| Milca → todos | data/processed/prices.csv | Precios ajustados diarios del grupo elegido, una columna por ticker |
| Milca → todos | data/processed/splits.json | Fechas de inicio y fin de train, test y valid (60/20/20 cronológico) |
| Milca → Arturo | artifacts/cointegration/selected_group.json | tickers, número de relaciones de cointegración, estadísticos de traza y de máximo eigenvalor, columnas del vector, vector elegido, pata de normalización y signo, p-value del ADF del spread, vida media en días |
| Milca | artifacts/cointegration/group_selection.csv | Todos los grupos probados, sus estadísticos, la etapa de descarte, la razón y el total evaluado |
| Milca | artifacts/cointegration/eigvec_rolling.csv | Primer eigenvector de Johansen móvil por fecha |
| Milca → Arturo y Paula | artifacts/kalman/beta_t.csv, spread_t.csv, gain_1.csv, posvar_1.csv | β̂_t por activo ya normalizado, spread S_t = β̂_tᵀP_t, ganancia de Kalman y varianza posterior del filtro 1 |
| Arturo → Paula | artifacts/kalman/gain_2.csv, posvar_2.csv, z_t.csv | Ganancia y varianza posterior del filtro 2; z-score con columnas z, mu_prev y sigma_prev |
| Arturo → Paula | strategy.generate_positions(z, params) | Señal diaria del spread en {-1, 0, +1} con entrada, salida, stop sobre abs(z), tenencia máxima y motivo de salida |
| Arturo → Paula | artifacts/strategy/params_default.json | z_entry, z_exit, z_stop y max_hold_days por defecto, calibrados a la vida media |
| Paula | artifacts/backtest/equity.csv, trades.csv, costs.csv, metrics.json, rolling_sharpe.csv, params.json, grid_search.csv | Curva de capital con drawdown y periodo; operaciones con pnl bruto, comisión, préstamo, pnl neto, retorno y motivo de salida; costos; métricas por periodo; umbrales finales y rejilla de calibración |
