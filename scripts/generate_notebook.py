"""Genera el cuaderno de Colab a partir de un esquema versionable."""

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


def markdown(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


notebook = nbf.v4.new_notebook()
notebook["metadata"] = {
    "colab": {"name": "Proyecto_Candelaria_Vial.ipynb", "provenance": []},
    "kernelspec": {"display_name": "Python 3", "name": "python3"},
    "language_info": {"name": "python"},
}

notebook["cells"] = [
    markdown(
        """
        # Candelaria Vial: priorización preventiva con datos abiertos

        **Asignatura:** Técnicas de Aprendizaje de Máquina  
        **Integrantes:** Completar nombres y códigos  
        **Fuente principal:** [Accidentalidad Vial Municipio de Candelaria, Valle](https://www.datos.gov.co/Transporte/Accidentalidad-Vial-Municipio-de-Candelaria-Valle-/7wbf-88zm)

        > **Nota de reproducibilidad.** Ejecutar el cuaderno de arriba hacia abajo y anotar la fecha de descarga: el conjunto abierto puede actualizarse.
        """
    ),
    markdown(
        """
        ## 1. Elevator pitch

        La Secretaría de Tránsito y Transporte de Candelaria debe decidir dónde focalizar controles, señalización, inspecciones y educación vial. Decidir por intuición o por el accidente más reciente puede dispersar recursos escasos. Nuestra propuesta usa los registros históricos de siniestros para estimar el volumen mensual esperado por corregimiento y estudiar, con cautela, patrones de severidad por lugar y momento.

        **Stakeholder:** Secretaría de Tránsito y Transporte de Candelaria, Valle del Cauca.  
        **Costo de inacción:** intervenciones tardías o mal focalizadas; costos humanos, de atención y de movilidad.  
        **Criterio de éxito:** superar un baseline temporal simple en el modelo mensual y no desplegar ningún modelo cuya validación temporal no sea discriminativa.

        ### Hipótesis

        1. La frecuencia mensual de siniestros no es homogénea entre corregimientos.
        2. Calendario, hora y ubicación aportan información sobre la severidad registrada.
        3. Los siniestros recientes mejoran la priorización mensual frente al promedio histórico único.
        """
    ),
    markdown(
        """
        ## 2. IA como revisora crítica: definición del problema

        **Prompt usado:**

        > Actúa como directora de movilidad de Candelaria. Queremos priorizar intervenciones de seguridad vial con registros históricos de siniestros por vía, corregimiento, fecha, hora y gravedad. Dame cinco razones concretas para no financiar este proyecto, qué evidencia pedirías y qué daño podría causar un falso positivo o falso negativo.

        **Respuesta crítica sintetizada:** (1) los registros pueden tener subregistro y cambios administrativos; (2) la vía y el corregimiento no prueban causalidad; (3) una predicción puede desplazar controles hacia barrios ya vigilados; (4) un modelo agregado no identifica conductas individuales; (5) sin comparación temporal con una regla simple, ML puede ser costo sin beneficio. Pediría cobertura por año, un baseline, evaluación fuera de tiempo y un protocolo de uso que prohíba sanciones o perfiles personales.

        **Acciones del equipo:** se separa el último año para prueba, se agregan meses con cero siniestros al panel de regresión, se compara contra un baseline, se excluye `clase_de_accidente` del predictor preventivo y se define la app como apoyo a inspección, nunca como mecanismo sancionatorio.
        """
    ),
    code(
        """
        # Instalación para Google Colab (ejecutar solo una vez)
        !pip -q install pandas numpy scikit-learn seaborn plotly

        import warnings
        from datetime import datetime, timezone

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.compose import ColumnTransformer
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.metrics import (
            accuracy_score, average_precision_score, confusion_matrix, f1_score,
            mean_absolute_error, mean_squared_error, precision_score, r2_score,
            recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", palette="deep")
        RANDOM_STATE = 42
        DATA_URL = "https://www.datos.gov.co/resource/7wbf-88zm.csv?$limit=100000"
        DOWNLOAD_TIME = datetime.now(timezone.utc).isoformat()
        """
    ),
    code(
        """
        # Descarga reproducible desde la API Socrata
        raw = pd.read_csv(DATA_URL)
        expected_columns = {
            "clase_de_accidente", "gravedad", "vias", "corregimiento",
            "d_a_semana", "fecha_de_ocurrecia", "hora_ocurrencia",
        }
        assert expected_columns.issubset(raw.columns), "El esquema publicado cambió; revisar el diccionario."
        print(f"Fecha UTC de descarga: {DOWNLOAD_TIME}")
        print(f"Filas: {len(raw):,}; columnas: {raw.shape[1]}")
        raw.head()
        """
    ),
    markdown(
        """
        ## 3. Diccionario y calidad de los datos

        | Campo | Tipo publicado | Uso |
        | --- | --- | --- |
        | `clase_de_accidente` | Texto | EDA; se excluye del predictor preventivo por posible fuga de información. |
        | `gravedad` | Texto | Objetivo: daños, heridos o muertos. |
        | `vias`, `corregimiento` | Texto | Contexto espacial de la priorización. |
        | `d_a_semana`, `fecha_de_ocurrecia`, `hora_ocurrencia` | Texto | Se transforman a calendario y hora. |

        La fuente es un registro administrativo. Por ello, un cero puede significar ausencia de reporte, no necesariamente ausencia absoluta de siniestros.
        """
    ),
    code(
        """
        quality = pd.DataFrame({
            "tipo": raw.dtypes.astype(str),
            "nulos": raw.isna().sum(),
            "n_unicos": raw.nunique(dropna=True),
        })
        display(quality)
        display(raw.duplicated().value_counts().rename("filas"))
        """
    ),
    code(
        """
        def clean_text(series):
            return (series.fillna("SIN_DATO").astype(str).str.strip().str.upper()
                    .replace({"": "SIN_DATO", "NAN": "SIN_DATO"}))

        incidents = raw.copy()
        for column in ["vias", "corregimiento", "d_a_semana", "gravedad", "clase_de_accidente"]:
            incidents[column] = clean_text(incidents[column])
        incidents["fecha"] = pd.to_datetime(incidents["fecha_de_ocurrecia"], dayfirst=True, errors="coerce")
        parsed_time = pd.to_timedelta(incidents["hora_ocurrencia"], errors="coerce")
        incidents["hora"] = (parsed_time.dt.total_seconds() / 3600).fillna(12).clip(0, 23.99)
        incidents = incidents.dropna(subset=["fecha"]).copy()
        incidents["anio"] = incidents.fecha.dt.year.astype(int)
        incidents["mes"] = incidents.fecha.dt.month.astype(int)
        incidents["dia_semana_num"] = incidents.fecha.dt.dayofweek.astype(int)
        incidents["es_fin_de_semana"] = (incidents.dia_semana_num >= 5).astype(int)
        incidents["mes_seno"] = np.sin(2 * np.pi * incidents.mes / 12)
        incidents["mes_coseno"] = np.cos(2 * np.pi * incidents.mes / 12)
        incidents["hora_seno"] = np.sin(2 * np.pi * incidents.hora / 24)
        incidents["hora_coseno"] = np.cos(2 * np.pi * incidents.hora / 24)
        incidents["lesion_o_muerte"] = incidents.gravedad.isin(["HERIDOS", "MUERTOS"]).astype(int)

        display(incidents[["fecha", "hora", "gravedad", "vias", "corregimiento"]].head())
        print("Cobertura:", incidents.fecha.min().date(), "a", incidents.fecha.max().date())
        """
    ),
    markdown(
        """
        ## 4. EDA con narrativa

        Cada gráfica responde a una decisión de negocio: ¿la cobertura es comparable por año?, ¿dónde conviene revisar primero?, ¿qué periodos conviene considerar en el plan preventivo? No se deben convertir correlaciones o conteos en afirmaciones de causalidad.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        sns.countplot(data=incidents, x="anio", hue="gravedad", ax=axes[0])
        axes[0].set_title("¿La composición de severidad es comparable entre años?")
        axes[0].set_xlabel("Año del siniestro")
        sns.countplot(data=incidents, x="mes", hue="gravedad", ax=axes[1])
        axes[1].set_title("¿Qué meses concentran los registros por gravedad?")
        axes[1].set_xlabel("Mes")
        plt.tight_layout()
        plt.show()

        top_locations = incidents.corregimiento.value_counts().head(10).sort_values()
        ax = top_locations.plot.barh(figsize=(9, 5), title="¿Qué corregimientos requieren revisión de capacidad preventiva?")
        ax.set_xlabel("Siniestros registrados")
        plt.show()
        """
    ),
    code(
        """
        pivot = pd.crosstab(incidents["hora"].astype(int), incidents["dia_semana_num"], normalize="columns")
        plt.figure(figsize=(10, 7))
        sns.heatmap(pivot, cmap="YlOrRd", cbar_kws={"label": "Proporción dentro del día"})
        plt.title("¿Qué franjas deben revisarse junto con los controles de campo?")
        plt.xlabel("Día de la semana (0=Lunes)")
        plt.ylabel("Hora")
        plt.show()

        severity_year = pd.crosstab(incidents.anio, incidents.gravedad, normalize="index").round(3)
        display(severity_year)
        """
    ),
    markdown(
        """
        **Hallazgo inesperado a discutir.** Si la proporción de `DAÑOS`, `HERIDOS` o `MUERTOS` cambia abruptamente entre años, no se debe atribuir de inmediato a una mejora o deterioro vial: puede representar un cambio de registro. Esta posibilidad afecta sobre todo el modelo de severidad y debe comprobarse con la entidad publicadora.
        """
    ),
    code(
        """
        # Panel completo corregimiento-mes: evita entrenar solo con meses que tuvieron siniestros.
        start, end = incidents.fecha.min().to_period("M"), incidents.fecha.max().to_period("M")
        periods = pd.period_range(start, end, freq="M")
        locations = pd.DataFrame({"corregimiento": sorted(incidents.corregimiento.unique())})
        monthly = locations.merge(pd.DataFrame({"periodo": periods}), how="cross")
        monthly["fecha"] = monthly.periodo.dt.to_timestamp()
        observed = (incidents.assign(periodo=incidents.fecha.dt.to_period("M"))
                    .groupby(["corregimiento", "periodo"], as_index=False).size()
                    .rename(columns={"size": "siniestros_mes"}))
        monthly = monthly.merge(observed, on=["corregimiento", "periodo"], how="left")
        monthly["siniestros_mes"] = monthly.siniestros_mes.fillna(0).astype(int)
        monthly = monthly.sort_values(["corregimiento", "fecha"]).reset_index(drop=True)
        groups = monthly.groupby("corregimiento", observed=True).siniestros_mes
        monthly["siniestros_lag_1"] = groups.shift(1).fillna(0)
        monthly["promedio_3_meses_previo"] = groups.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
        monthly["anio"] = monthly.fecha.dt.year.astype(int)
        monthly["mes"] = monthly.fecha.dt.month.astype(int)
        monthly["mes_seno"] = np.sin(2 * np.pi * monthly.mes / 12)
        monthly["mes_coseno"] = np.cos(2 * np.pi * monthly.mes / 12)
        print(monthly.shape)
        monthly.head()
        """
    ),
    markdown(
        """
        ## 5. Preparación y partición temporal

        Usamos el último año disponible solo para prueba. Los lags para la regresión se calculan con periodos anteriores. `clase_de_accidente`, muertes y cualquier consecuencia posterior no se usan como predictores preventivos.
        """
    ),
    code(
        """
        test_year = int(incidents.anio.max())
        inc_train, inc_test = incidents[incidents.anio < test_year].copy(), incidents[incidents.anio == test_year].copy()
        mon_train, mon_test = monthly[monthly.anio < test_year].copy(), monthly[monthly.anio == test_year].copy()
        print(f"Entrenamiento: hasta {test_year - 1}; prueba: {test_year}")
        print("Incidentes train/test:", inc_train.shape, inc_test.shape)
        print("Panel mensual train/test:", mon_train.shape, mon_test.shape)

        def preprocessor(categorical, numeric):
            return ColumnTransformer([
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                                  ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), categorical),
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                                  ("scale", StandardScaler())]), numeric),
            ])
        """
    ),
    markdown(
        """
        ## 6. Problema 1 — regresión de siniestros mensuales

        La salida es una cantidad esperada de siniestros en el corregimiento durante el mes. Se comparan tres algoritmos cubiertos por el curso y un baseline de promedio histórico. En una siguiente iteración, hacer `RandomizedSearchCV` usando 2024 como validación dentro del entrenamiento; no tocar el año de prueba.
        """
    ),
    code(
        """
        reg_cat = ["corregimiento"]
        reg_num = ["anio", "mes", "mes_seno", "mes_coseno", "siniestros_lag_1", "promedio_3_meses_previo"]
        reg_features = reg_cat + reg_num
        reg_models = {
            "Baseline: promedio histórico": DummyRegressor(strategy="mean"),
            "Ridge": Ridge(alpha=8.0),
            "Árbol de decisión": DecisionTreeRegressor(max_depth=6, min_samples_leaf=8, random_state=RANDOM_STATE),
            "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=4,
                                                     max_features=0.8, n_jobs=-1, random_state=RANDOM_STATE),
        }
        reg_fitted, reg_rows = {}, []
        for name, estimator in reg_models.items():
            pipe = Pipeline([("prep", preprocessor(reg_cat, reg_num)), ("model", estimator)])
            pipe.fit(mon_train[reg_features], mon_train.siniestros_mes)
            pred = pipe.predict(mon_test[reg_features])
            reg_rows.append({"Modelo": name, "MAE": mean_absolute_error(mon_test.siniestros_mes, pred),
                             "RMSE": mean_squared_error(mon_test.siniestros_mes, pred) ** 0.5,
                             "R2": r2_score(mon_test.siniestros_mes, pred)})
            reg_fitted[name] = pipe
        reg_results = pd.DataFrame(reg_rows).sort_values("MAE")
        display(reg_results.round(4))
        """
    ),
    markdown(
        """
        **Interpretación requerida:** compare la mejora absoluta de MAE frente al baseline. Un R² bajo no debe ocultarse: señala que faltan variables relevantes, como flujo vehicular, clima, obras, controles y exposición de actores viales. La salida solo sirve para orientar una revisión humana del territorio.
        """
    ),
    markdown(
        """
        ## 7. Problema 2 — clasificación de lesión o muerte

        La etiqueta es `HERIDOS` o `MUERTOS` frente a `DAÑOS`. Esta tarea es exploratoria: si la composición de gravedad cambia entre años o el AUC no supera el baseline, no se lleva a la app. Reportar F1 **y** AUC-ROC/AUC-PR, pues accuracy y F1 pueden verse altos cuando una clase domina el año de prueba.
        """
    ),
    code(
        """
        clf_cat = ["vias", "corregimiento"]
        clf_num = ["anio", "mes", "dia_semana_num", "hora", "es_fin_de_semana", "mes_seno", "mes_coseno", "hora_seno", "hora_coseno"]
        clf_features = clf_cat + clf_num
        clf_models = {
            "Baseline: clase mayoritaria": DummyClassifier(strategy="prior"),
            "Regresión logística": LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000,
                                                        solver="liblinear", random_state=RANDOM_STATE),
            "Árbol de decisión": DecisionTreeClassifier(max_depth=6, min_samples_leaf=12,
                                                         class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(n_estimators=350, max_depth=12, min_samples_leaf=6,
                                                      max_features=0.8, class_weight="balanced_subsample",
                                                      n_jobs=-1, random_state=RANDOM_STATE),
        }
        clf_fitted, clf_rows = {}, []
        for name, estimator in clf_models.items():
            pipe = Pipeline([("prep", preprocessor(clf_cat, clf_num)), ("model", estimator)])
            pipe.fit(inc_train[clf_features], inc_train.lesion_o_muerte)
            proba = pipe.predict_proba(inc_test[clf_features])[:, 1]
            label = (proba >= 0.5).astype(int)
            clf_rows.append({"Modelo": name, "Accuracy": accuracy_score(inc_test.lesion_o_muerte, label),
                             "Precision": precision_score(inc_test.lesion_o_muerte, label, zero_division=0),
                             "Recall": recall_score(inc_test.lesion_o_muerte, label, zero_division=0),
                             "F1": f1_score(inc_test.lesion_o_muerte, label, zero_division=0),
                             "AUC-ROC": roc_auc_score(inc_test.lesion_o_muerte, proba),
                             "AUC-PR": average_precision_score(inc_test.lesion_o_muerte, proba)})
            clf_fitted[name] = (pipe, proba)
        clf_results = pd.DataFrame(clf_rows).sort_values("AUC-ROC", ascending=False)
        display(clf_results.round(4))

        best_name = clf_results.iloc[0].Modelo
        best_pipe, best_proba = clf_fitted[best_name]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        sns.heatmap(confusion_matrix(inc_test.lesion_o_muerte, best_proba >= .5), annot=True, fmt="d", ax=axes[0])
        axes[0].set_title(f"Matriz de confusión: {best_name}")
        RocCurveDisplay.from_predictions(inc_test.lesion_o_muerte, best_proba, ax=axes[1])
        PrecisionRecallDisplay.from_predictions(inc_test.lesion_o_muerte, best_proba, ax=axes[2])
        plt.tight_layout()
        plt.show()
        """
    ),
    markdown(
        """
        ## 8. IA como revisora crítica: selección del modelo

        **Prompt usado:**

        > En la prueba temporal, la clasificación tiene F1 alto, pero AUC-ROC cercano a 0,5 y la clase de lesión o muerte domina el último año. ¿Es responsable seleccionar este modelo? ¿Qué explicaciones alternativas y decisiones recomendarías?

        **Respuesta crítica sintetizada:** No. El F1 puede ser alto porque predecir siempre la clase mayoritaria funciona en ese corte. El cambio de composición puede ser un cambio de registro o de distribución. Recomienda mostrar baseline, AUC-ROC y AUC-PR, investigar calidad con la entidad, no publicar el clasificador y mantener la app limitada al problema mensual si este supera su baseline.

        **Decisión del equipo:** se documenta el clasificador y sus límites porque es un requisito académico, pero se bloquea su uso en la app si AUC-ROC < 0,70. Se solicita validar con la entidad el cambio histórico de la categoría `DAÑOS`.
        """
    ),
    markdown(
        """
        ## 9. Ética, sesgos y limitaciones

        - Los datos reflejan lo que se registra; no equivalen necesariamente a todos los siniestros.
        - Una mayor frecuencia registrada puede reflejar mejor reporte o vigilancia, no mayor riesgo real.
        - Vías y corregimientos pueden ser proxies de desigualdad. El producto debe asignar **apoyo preventivo**, nunca justificar estigmas, sanciones individuales o controles discriminatorios.
        - Los lags requieren que la entidad actualice los conteos mensuales antes de usar la app.
        - La herramienta no descubre causas ni responsabilidad: una inspección técnica y participación local son indispensables.
        """
    ),
    markdown(
        """
        ## 10. Conclusiones y trabajo futuro

        El proyecto ofrece una ruta reproducible para priorizar inspecciones mensuales. La decisión de desplegar depende de la validación temporal, no de una métrica aislada. Próximos pasos: normalizar nombres de vías, conseguir exposición (aforo vehicular, peatones y ciclistas), clima, obras, controles y datos geográficos; validar los hallazgos con comunidad y Secretaría; medir la efectividad real de las intervenciones con un diseño cuasiexperimental.
        """
    ),
]

output = Path("notebooks/Proyecto_Candelaria_Vial.ipynb")
output.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, output)
print(f"Cuaderno creado: {output}")
