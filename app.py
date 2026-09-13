"""Aplicación para apoyar la prevención vial en Candelaria, Valle."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from src.features import regression_features

st.set_page_config(page_title="Candelaria Vial", page_icon="🚦", layout="wide")
ARTIFACT_DIR = Path("artifacts")


@st.cache_resource
def load_models():
    return (
        joblib.load(ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib"),
        json.loads((ARTIFACT_DIR / "metricas.json").read_text(encoding="utf-8")),
    )


def recommended_actions(count: float) -> list[str]:
    actions = []
    if count >= 1:
        actions.append("Priorizar una inspección de señalización, iluminación y puntos de conflicto en el corregimiento.")
    else:
        actions.append("Mantener monitoreo mensual y contrastar la estimación con reportes de la comunidad.")
    actions.append("Validar en terreno antes de intervenir: el modelo prioriza, no demuestra causalidad ni reemplaza el criterio técnico.")
    return actions


st.title("🚦 Candelaria Vial")
st.caption("Priorización preventiva agregada para la Secretaría de Tránsito y Transporte de Candelaria, Valle.")

required = ["modelo_priorizacion_mensual.joblib", "metricas.json"]
if not all((ARTIFACT_DIR / name).exists() for name in required):
    st.error("Aún no hay modelos entrenados. Ejecuta `python -m src.train` antes de publicar la aplicación.")
    st.stop()

monthly_model, metadata = load_models()
st.info(
    f"Fuente: {metadata['registros_incidentes']:,} siniestros registrados entre "
    f"{metadata['fecha_minima']} y {metadata['fecha_maxima']}. La prueba temporal usa {metadata['anio_prueba']}."
)

st.subheader("Prioriza el próximo mes")
corregimiento = st.selectbox("Corregimiento", metadata["corregimientos"])
target_date = st.date_input("Mes que se quiere priorizar", value=pd.Timestamp("2026-01-01"))
previous = st.number_input("Siniestros registrados el mes anterior", min_value=0, value=0, step=1)
rolling = st.number_input("Promedio de siniestros de los tres meses previos", min_value=0.0, value=0.0, step=0.1)
target_month = pd.Timestamp(target_date).month
monthly_row = pd.DataFrame({
    "corregimiento": [corregimiento], "anio": [pd.Timestamp(target_date).year],
    "mes": [target_month], "mes_seno": [np.sin(2 * np.pi * target_month / 12)],
    "mes_coseno": [np.cos(2 * np.pi * target_month / 12)],
    "siniestros_lag_1": [previous], "promedio_3_meses_previo": [rolling],
})
reg_categorical, reg_numeric = regression_features()
expected_count = max(0.0, float(monthly_model.predict(monthly_row[reg_categorical + reg_numeric])[0]))
st.metric("Siniestros estimados", f"{expected_count:.2f}")

st.subheader("Acciones sugeridas")
for action in recommended_actions(expected_count):
    st.write(f"- {action}")

with st.expander("Cómo interpretar este resultado"):
    st.write(
        "La estimación se refiere a la cantidad mensual esperada de siniestros por corregimiento. "
        "No estima la probabilidad de que una persona específica tenga un accidente ni identifica causas."
    )

st.subheader("Transparencia del modelo")
factors = pd.DataFrame(metadata.get("factores_importantes_regresion", []))
if factors.empty:
    st.write("El modelo seleccionado no expone importancias globales comparables.")
else:
    st.caption("Variables con mayor contribución relativa al modelo; no prueban causalidad.")
    st.dataframe(factors, hide_index=True, use_container_width=True)

if not metadata.get("clasificacion_apta_para_despliegue", False):
    st.warning(
        "La clasificación de severidad se documenta en el informe, pero no se publica en esta app: "
        "su validación temporal no alcanzó el umbral de discriminación definido por el equipo."
    )
    st.write(
        "Los registros son administrativos y pueden tener subregistro, cambios en los mecanismos de reporte y "
        "una representación desigual de vías o poblaciones. No use la app para sancionar a personas, atribuir "
        "culpa ni sustituir una inspección técnica."
    )
