# Candelaria Vial: priorización preventiva con datos abiertos

Proyecto de Técnicas de Aprendizaje de Máquina — Pontificia Universidad Javeriana.

## Elevator pitch

**Problema real.** La Secretaría de Tránsito y Transporte de Candelaria (Valle) debe decidir cada mes dónde concentrar controles, señalización, inspecciones y educación vial. Decidir solo por intuición o por el último accidente puede dispersar recursos escasos y dejar sin atención los patrones recurrentes por corregimiento, vía, fecha y hora.

**Stakeholder.** Secretaría de Tránsito y Transporte de Candelaria, Valle del Cauca. La app está diseñada como insumo de priorización para su equipo de seguridad vial, no para sancionar a personas ni establecer culpabilidades.

**Costo de inacción.** Sin una priorización consistente, las intervenciones llegan tarde o se asignan a zonas de alta visibilidad y no necesariamente de mayor necesidad. Esto puede sostener siniestros con daños humanos, costos de atención, pérdida de productividad y deterioro de la confianza ciudadana.

**Solución.** Con registros históricos de siniestros del municipio, el proyecto estima (1) cuántos siniestros pueden registrarse en un corregimiento durante un mes y (2) si, dado un siniestro, las condiciones de lugar y tiempo se parecen a casos con lesionados o fallecidos. La app traduce ambas señales en acciones preventivas.

## Fuente principal y reproducibilidad

`Accidentalidad Vial Municipio de Candelaria, Valle`, publicado por la Alcaldía de Candelaria en Datos Abiertos Colombia. Tiene registros por siniestro con clase, gravedad, vía, corregimiento, día, fecha y hora.

- Página del conjunto: <https://www.datos.gov.co/Transporte/Accidentalidad-Vial-Municipio-de-Candelaria-Valle-/7wbf-88zm>
- API reproducible: <https://www.datos.gov.co/resource/7wbf-88zm.csv?$limit=100000>
- Identificador Socrata: `7wbf-88zm`.

La primera ejecución guarda una copia en `data/raw/`, que está excluida de Git porque la fuente pública puede actualizarse. En el cuaderno deben registrar fecha/hora de descarga, número de filas y fecha máxima observada.

## Hipótesis a contrastar

1. La frecuencia de siniestros presenta diferencias persistentes entre corregimientos y meses; por tanto, una tasa promedio municipal no es suficiente para priorizar recursos.
2. La actividad vial cambia con la franja horaria y el día de la semana, y estos patrones aportan información para distinguir siniestros que terminan en lesión o muerte de los que solo generan daños.
3. El número de siniestros de los meses previos, junto con calendario y corregimiento, predice mejor el volumen mensual que una regla plana basada en el promedio histórico.

Son hipótesis, no conclusiones. El EDA debe mostrar evidencia a favor o en contra, incluyendo tamaños de muestra y años comparables.

## Por qué ML y cuándo no usarlo

Un dashboard responde dónde y cuándo ocurrió la accidentalidad. La necesidad operativa adicional es priorizar el próximo ciclo combinando lugar, estacionalidad y comportamiento reciente, y estimar la severidad probable de distintos escenarios. La evaluación es temporal —el último año queda intacto— para no confundir patrones futuros con información ya vista.

Sin embargo, si los modelos no superan un baseline simple, varían mucho entre años o no dan una ventaja interpretativa que justifique sus errores, la recomendación final debe ser un dashboard descriptivo y una regla de priorización transparente. ML no es un requisito que deba sobrevivir a mala evidencia.

## Problemas y comparación de modelos

**Correcciones de esta entrega.** El cuaderno y el entrenamiento reproducible incluyen tendencia central y dispersión, asimetría, correlaciones, interacciones documentadas con IA por fase y optimización temporal de hiperparámetros sin usar el año de prueba para seleccionar modelos.

| Problema | Unidad | Objetivo | Algoritmos | Métricas |
| --- | --- | --- | --- | --- |
| Regresión | Corregimiento-mes | `siniestros_mes` | Ridge, árbol de decisión, Random Forest | MAE, RMSE, R² |
| Clasificación | Siniestro | `lesion_o_muerte` (`HERIDOS` o `MUERTOS`) | regresión logística, árbol de decisión, Random Forest | Accuracy, precision, recall, F1, AUC-ROC, AUC-PR |

La partición temporal usa todos los años anteriores para entrenar y el último año disponible para probar. La tarea de regresión crea explícitamente el panel completo de corregimiento-mes; así incluye meses sin accidentes en vez de aprender solo de los meses con casos.

### Diccionario y justificación de variables

| Campo / variable | Tipo | Uso y justificación |
| --- | --- | --- |
| `vias` | Categórica | Vía registrada; contexto espacial del escenario preventivo. |
| `corregimiento` | Categórica | Unidad territorial en la que el stakeholder puede focalizar una acción. |
| `fecha_de_ocurrecia`, `hora_ocurrencia`, `d_a_semana` | Fecha/hora/texto | Se usan para derivar año, mes, día, hora, fin de semana y variables cíclicas. Se verifica la consistencia entre fecha y día publicado. |
| `gravedad` | Categórica | Objetivo de clasificación; se transforma en lesión o muerte vs. solo daños. |
| `siniestros_lag_1`, `promedio_3_meses_previo` | Numérica derivada | Solo para el modelo mensual; resumen de meses anteriores, sin mirar el futuro. |
| `clase_de_accidente` | Categórica | Se explora en EDA pero se excluye del predictor preventivo: se conoce con el siniestro y puede introducir fuga de información. |

## Ejecución

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.train
streamlit run app.py
```

El entrenamiento descarga la fuente, limpia los datos, compara tres algoritmos por problema y guarda el modelo ganador y métricas reales en `artifacts/`. La app exige dichos artefactos para iniciar.

## Estructura del cuaderno de Colab

1. Elevator pitch, stakeholder, costo de inacción, hipótesis y criterio de éxito.
2. Origen de datos y diccionario: descargar desde la API Socrata, registrar versión y auditar tipos.
3. EDA con narrativa: cobertura por año, calidad de fecha/hora, gravedad, vías/corregimientos, mes-hora y días; explicar qué decisión pública informa cada gráfica.
4. Hallazgos inesperados y sesgos: cambios de cobertura por año, vías mal estandarizadas, baja muestra en ubicaciones raras y subregistro.
5. Preparación e ingeniería: normalización de texto, fechas, variables cíclicas, panel de ceros, lags sin fuga y separación temporal.
6. Regresión: baseline, tres modelos, ajuste de hiperparámetros, métricas temporales y análisis de errores.
7. Clasificación: baseline, tres modelos, matriz de confusión, ROC y precision-recall; elegir umbral según el costo de perder un escenario grave frente a una falsa alarma.
8. Iteración: contrastar versión sin lags vs. con lags en regresión y versión sin calendario cíclico vs. con él en clasificación; conservar solo mejoras defendibles.
9. Ética, límites, conclusión y trabajo futuro.

## Uso crítico de IA que se debe documentar

Incluyan el prompt completo, la respuesta, la decisión del equipo y la evidencia que la sostiene. Ejemplos:

> Actúa como directora de movilidad de Candelaria. Queremos priorizar intervenciones de seguridad vial con registros históricos de siniestros por vía, corregimiento, fecha, hora y gravedad. Dame cinco razones concretas para no financiar este proyecto, qué evidencia pedirías y qué daño podría causar un falso positivo o falso negativo.

> Queremos usar Random Forest para predecir siniestros mensuales y lesión o muerte. ¿Qué riesgos de fuga de información, cambio de distribución temporal y uso indebido aparecen? Propón un criterio medible para decidir que un dashboard descriptivo es preferible al ML.

La respuesta de IA no es evidencia; sirve para cuestionar decisiones. La decisión final debe justificar si se acepta, se ajusta o se refuta la crítica.

## Ética, sesgos y limitaciones

- El conjunto registra eventos reportados, no todos los eventos que ocurrieron. Un aumento puede reflejar tanto riesgo como mejor registro.
- Cobertura y categorías pueden cambiar entre años; evaluar sobre el último año expone parte de este problema, pero no lo elimina.
- Vía y corregimiento pueden correlacionarse con desigualdad de infraestructura y vigilancia. El resultado debe asignar recursos preventivos adicionales, nunca estigmatizar barrios ni justificar controles discriminatorios.
- La clasificación se interpreta **condicionada a que haya un siniestro**; no es una probabilidad individual de accidente.
- La app no identifica causas ni responsabilidad. Toda intervención requiere validación de campo y participación local.

## Despliegue

Para Streamlit Community Cloud, subir `app.py`, `src/`, `requirements.txt` y los artefactos entrenados en `artifacts/` (sin datos crudos). El punto de entrada es `app.py`. En Hugging Face Spaces, seleccionar Streamlit y usar los mismos archivos.
