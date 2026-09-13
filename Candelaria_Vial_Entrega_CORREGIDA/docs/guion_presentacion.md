# Guion de presentación — Candelaria Vial (20 minutos)

Este guion debe acompañarse de visualizaciones generadas en el cuaderno, no de capturas genéricas. Reemplazar los nombres de integrantes antes de presentar.

| Minutos | Diapositiva | Mensaje central | Evidencia o demostración |
| --- | --- | --- | --- |
| 0:00–0:45 | 1. Candelaria Vial | No queremos “predecir accidentes” por moda: queremos decidir dónde revisar primero. | Una frase del problema y una foto/mapa contextual con crédito. |
| 0:45–2:00 | 2. Decisión pública | El stakeholder es la Secretaría de Tránsito; su recurso escaso es capacidad preventiva. | Antes/después conceptual: priorización por intuición → lista mensual con evidencia. |
| 2:00–3:30 | 3. Datos y alcance | Se analizaron 1.208 siniestros de 2021 a 2025 de Datos Abiertos Colombia. | Ficha del conjunto, granularidad y columnas; aclarar que es registro administrativo. |
| 3:30–5:00 | 4. Qué encontramos | La cobertura y la composición de gravedad cambian entre años; ese hallazgo condiciona el uso responsable del ML. | Barras de gravedad por año y top de corregimientos del cuaderno. |
| 5:00–6:30 | 5. Problema 1 | Convertimos eventos en un panel completo de corregimiento-mes, incluyendo ceros y lags sin mirar el futuro. | Diagrama: eventos → panel mensual → entrenamiento hasta 2024 / prueba 2025. |
| 6:30–8:30 | 6. Resultado de regresión | Random Forest redujo MAE de 0,717 (baseline) a 0,445: mejora aproximada de 38%, con R² = 0,139. Es una señal de priorización, no una promesa causal. | Tabla de métricas y un ejemplo de una fila de la app. |
| 8:30–10:00 | 7. Problema 2 | Clasificamos lesión o muerte con tres algoritmos, pero la prueba temporal fue débil: AUC-ROC máximo 0,567, apenas sobre 0,500 del baseline. | Tabla de clasificación, matriz de confusión y gráfica de cambio de clases. |
| 10:00–11:30 | 8. Decisión técnica | No desplegamos el clasificador de severidad. Preferimos una aplicación más limitada pero honesta. | Regla explícita: no publicar clasificación si AUC-ROC < 0,70. |
| 11:30–13:30 | 9. Demo de la app | El usuario elige corregimiento, mes y dos indicadores recientes; recibe número estimado, acciones y factores globales. | Demostración de [app.py](../app.py) o video corto de respaldo. |
| 13:30–15:00 | 10. Ética y límites | No usar para culpar, sancionar ni perfilar personas; validar en campo y atender posible subregistro. | Cuadro “sí sirve para / no sirve para”. |
| 15:00–16:00 | 11. Siguiente paso y solicitud | Pedimos a la Secretaría validar la calidad histórica y aportar aforos, clima, obras y controles. | Lista priorizada de datos que mejoran el modelo. |
| 16:00–20:00 | Preguntas | Defender decisiones y límites, no solo algoritmos. | Mantener a la vista la tabla de métricas y el diagrama de flujo. |

## Preguntas difíciles y respuestas breves

**¿Por qué ML y no un dashboard?**

El dashboard es necesario y muestra lo ocurrido. El modelo mensual se evaluó contra un promedio histórico y redujo el MAE. Aun así, si no mejora con nuevos datos, recomendamos conservar el dashboard y la regla transparente.

**¿Por qué el F1 del clasificador es alto pero no lo usan?**

En 2025 la clase lesión o muerte domina el conjunto. Predecir siempre esa clase tiene F1 alto. AUC-ROC mide si el modelo separa casos y fue solo 0,567; ocultar esa debilidad sería una mala práctica.

**¿Qué acción se deriva de una estimación alta?**

No una sanción automática. Se prioriza una inspección de iluminación, señalización y conflictos viales; después se contrasta el hallazgo con comunidad y equipo técnico.

**¿El modelo prueba la causa de un siniestro?**

No. Estima patrones de registros pasados. Para causalidad se requieren variables de exposición y un diseño de evaluación de intervenciones.
