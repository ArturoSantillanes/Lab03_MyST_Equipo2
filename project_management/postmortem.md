# Post-mortem

> **Borrador generado con asistencia de IA a partir del historial del proyecto. Pendiente de revisión por los tres integrantes.**

## Qué hitos se cumplieron a tiempo y cuáles se retrasaron (y por qué)

Las fechas reales son las de los commits, y eso marca cuándo se subió el trabajo, no cuándo se hizo. Los commits de Paula quedaron con la fecha en que se reescribió el autor (`handoff_paula.md`), así que para M5 y M6 solo se sabe la fecha máxima.

| Hito | Plan | Real | Resultado |
|---|---|---|---|
| M0 Repositorio | 26 oct | 26 oct | A tiempo |
| M1 Datos | 26 oct | 28 oct (oro); 29 oct (v5) | Retraso de 2 a 3 días |
| M2 Cointegración | 27 oct | 28 oct (oro); 29 oct (eléctricas) | Retraso de 1 a 2 días |
| M3 Filtro 1 | 27 oct | 28 oct; versión final el 29 oct | Retraso de 1 a 2 días |
| M4 Filtro 2, VECM, señales | 29 oct | 29 oct, 23:55 | A tiempo, sin margen; corrió en Windows hasta el 30 oct |
| M5 y M6 Backtest y dashboard | 31 oct | A más tardar el 1 nov | Posible retraso de hasta 1 día |
| M7 Integración, reporte, README | 1 nov | Integración y README el 1 nov; reporte el 3 nov | Parcial: el reporte se retrasó 2 días |
| M8 Último commit | 2 nov | No se cumplió | La fecha real de entrega resultó ser el 9 nov |
| M9 Colchón y ensayos | 3 nov | El 3 nov se usó para auditar y cerrar pendientes | No hubo colchón. El repositorio no registra ensayos |

La causa principal del retraso fue la búsqueda del grupo: cada versión descartada obligaba a repetir el embudo. Como el orden de trabajo era en cadena (Milca, Arturo, Paula), ese retraso se comprimió sobre los bloques siguientes y dejó el reporte y este documento para el final.

## Cuellos de botella técnicos y organizacionales

**Técnicos**

- **Búsqueda del grupo.** Hubo varias versiones: 9 productoras de petróleo (ninguna pasó la correlación), 2 × 8,008 grupos de petroleras, tres versiones de oro y la v5 con 8 sectores. El oro no alcanzaba la correlación de 0.7 y su estabilidad era de 21%. Salir de ahí exigió preregistrar sectores antes de descargar datos. El total llegó a 17,162 grupos, y esa cifra pesa contra el resultado: la traza del ganador no sobrevive a Bonferroni.
- **Intercepto α en el filtro 1.** El filtro final trabaja sin intercepto. El repositorio documenta la decisión, pero no el problema que la motivó.
- **Máxima verosimilitud para Q en el filtro 1.** Elegía una Q demasiado grande: los pesos perseguían cada desviación y se comían la reversión. Se cambió por una rejilla con criterios escritos antes y una meseta de q.
- **Encoding en Windows.** El pipeline se detenía en `data` porque el código se desarrolló en Mac y leía JSON sin `encoding="utf-8"`. Se detectó el 29 de octubre y se arregló el 30.
- **Tiempo de `cointegration`.** Mide de 37 a 39 minutos en Windows contra 6.5 en Mac. La causa probable son los hilos de OpenBLAS, sin comprobar. Cada verificación completa costó más de 40 minutos, y eso desincentivó repetirlas.
- **Costos contra señal.** Esto no se pudo resolver: en train, la comisión por operación ($2.0k) supera la ganancia bruta media ($1.5k). La regla de 30 operaciones obliga a umbrales bajos, justo donde la reversión por operación es más chica.

**Organizacionales**

- **Revisión cruzada que no ocurrió.** El charter pide que cada PR lo revise otra persona, pero los PR #1 y #2 se fusionaron con 0 revisiones en GitHub, el trabajo de Milca no tuvo un PR propio, y la integración final del 3 de noviembre tampoco tiene revisión registrada. La protección de `main` era solo un acuerdo.
- **Permisos de GitHub.** Paula no pudo subir desde su sesión por falta de permiso de escritura, y se resolvió agregándola como colaboradora. Al reescribir el autor de sus commits se perdieron las fechas reales.
- **Errores que se copian entre secciones.** Las tres secciones decían que las 3 sub-ventanas inestables son las últimas de train; en realidad son la 10, la 13 y la 14. El error nació en una sección y se copió a las demás sin volver al CSV. Es la misma falla del Lab 02, y se encontró hasta la auditoría final.
- **Uso de IA sin cerrar.** El README dice que cada quien revisó y entendió su código, mientras `ia_log.md` todavía marca esa revisión como pendiente en 8 filas.
- **Documentos que envejecen.** Los handoffs se escribieron una sola vez, y varios quedaron con fallas ya resueltas.

## Qué haríamos distinto en un entorno de producción real

- **Integración continua:** correr pruebas y un pipeline corto en Windows y Linux en cada PR. El error de encoding habría salido el primer día.
- **Revisión obligatoria:** proteger `main` en GitHub, para que un PR no se pueda fusionar sin aprobación.
- **Reporte generado desde los datos:** que tablas y cifras se lean de los artefactos en lugar de copiarse a mano, para evitar errores como el de las sub-ventanas.
- **Entorno bloqueado:** fijar también las dependencias indirectas y la versión de Python. Hoy se instala hasta una versión de desarrollo de `wrapt`.
- **Paper trading antes de arriesgar capital:** con estos resultados, esta estrategia no pasaría a producción. Ningún Sharpe se distingue de cero, y el Deflated Sharpe de train es ≈ 0.
- **Costos reales:** modelar spread, impacto de mercado y disponibilidad de préstamo; el backtest actual supone ejecución al cierre sin ellos, lo que favorece a la estrategia.
- **Monitoreo y salida por ruptura:** vigilar en vivo el ADF del spread y las innovaciones del filtro 2, y agregar un stop sobre el nivel del spread, porque el stop sobre |z| casi no protege.

## Cómo se materializaron (o no) los riesgos del registro

| # | Riesgo (probabilidad / impacto) | ¿Se materializó? | Qué pasó con la mitigación |
|---|---|---|---|
| 1 | Ningún grupo pasa Johansen y ADF (media / alto) | Sí, en las primeras versiones | Ampliar el universo funcionó y no se relajaron criterios: 4 de 910 grupos cumplen los 7 requisitos. El costo fue más pruebas múltiples |
| 2 | Pérdida de cointegración fuera de muestra (alta / alto) | Parcialmente | El bruto sigue positivo en test y valid, pero las innovaciones se deterioran en valid (varianza 1.394, Ljung-Box p = 0.027). Con los parámetros finales, el stop actuó 1 vez y la tenencia máxima 0. **El ADF del spread en test y valid que prometía el charter no se midió** |
| 3 | Filtro diverge o queda mal calibrado (media / alto) | Parcialmente | El filtro 1 se comía la reversión con máxima verosimilitud y se corrigió antes de usarlo. El filtro 2 pasa la prueba en train, pero no en valid; no se recalibró porque estaría prohibido |
| 4 | Falta de tiempo para dashboard y reporte (media / alto) | Sí, en el reporte | El dashboard se terminó, pero el reporte y este documento quedaron para el final y el colchón del 3 de noviembre se consumió |
| 5 | Commits o dominio desbalanceados (media / alto) | Parcialmente | Todos tienen commits (Milca 25, Paula 16, Arturo 11), pero la revisión cruzada no ocurrió y las revisiones personales siguen abiertas. Falta comprobar que los tres pueden explicar todo |
| 6 | Enfermedad o ausencia (baja / medio) | Sin registro | El repositorio no documenta ninguna ausencia |
