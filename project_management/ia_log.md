# Registro de uso de IA

Aquí se anota cada vez que se usa un asistente de IA: qué generó y cómo lo verificó la persona. Cada integrante es responsable de todo el código que entrega y debe poder explicarlo línea por línea. Se agrega una fila por uso.

| Fecha | Persona | Archivo o parte | Qué generó la IA | Cómo se verificó |
|---|---|---|---|---|
| 2026-10-26 | Arturo | Repositorio inicial y documentos de gestión | Borradores generados con Claude Code | Revisados archivo por archivo con el ciclo de verificación |
| 2026-10-29 | Milca | Bloque completo rehecho desde cero en `milca/cointegracion-kalman1`: data/ (universo en configuración, descarga, limpieza, partición), cointegration/ (embudo de 7 etapas, estabilidad, eigenvector móvil, figuras), kalman/filter1_weights.py, 4 pruebas, handoff y docs/secciones/milca.md | Código, figuras y documentos generados con Claude Code siguiendo el documento de instrucciones del equipo | Pipeline corrido de punta a punta; ejemplo numérico de 5 periodos recalculado a mano (diferencia 0 contra el código); prueba de un paso a mano; figuras revisadas con vistas recortadas a train; ciclo de verificación con lista de requisitos; se dejaron en NO CUMPLE (sin relajar) la etapa 2 y la prueba de estabilidad de signo; revisión personal de Milca pendiente |
