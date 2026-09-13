"""Limpieza y variables predictoras para la siniestralidad vial de Candelaria."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET_CLASSIFICATION = "lesion_o_muerte"
TARGET_REGRESSION = "siniestros_mes"


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("SIN_DATO")
        .astype(str)
        .str.strip()
        .str.upper()
        .replace({"": "SIN_DATO", "NAN": "SIN_DATO"})
    )


def prepare_incidents(raw: pd.DataFrame) -> pd.DataFrame:
    """Convierte el registro de siniestros en una tabla analítica por incidente.

    `clase_de_accidente` se conserva para el EDA pero se excluye de las variables
    predictoras: se determina junto con el siniestro y no es apropiada para una
    priorización preventiva previa.
    """
    data = raw.copy()
    for column in ["vias", "corregimiento", "d_a_semana", "gravedad", "clase_de_accidente"]:
        data[column] = _clean_text(data[column])

    data["fecha"] = pd.to_datetime(
        data["fecha_de_ocurrecia"], dayfirst=True, errors="coerce"
    )
    parsed_time = pd.to_timedelta(data["hora_ocurrencia"], errors="coerce")
    data["hora"] = (parsed_time.dt.total_seconds() / 3600).fillna(12).clip(0, 23.99)

    data = data.dropna(subset=["fecha"]).copy()
    data["anio"] = data["fecha"].dt.year.astype(int)
    data["mes"] = data["fecha"].dt.month.astype(int)
    data["dia_semana_num"] = data["fecha"].dt.dayofweek.astype(int)
    data["es_fin_de_semana"] = (data["dia_semana_num"] >= 5).astype(int)
    data["mes_seno"] = np.sin(2 * np.pi * data["mes"] / 12)
    data["mes_coseno"] = np.cos(2 * np.pi * data["mes"] / 12)
    data["hora_seno"] = np.sin(2 * np.pi * data["hora"] / 24)
    data["hora_coseno"] = np.cos(2 * np.pi * data["hora"] / 24)
    data[TARGET_CLASSIFICATION] = data["gravedad"].isin(["HERIDOS", "MUERTOS"]).astype(int)
    return data.reset_index(drop=True)


def classification_features() -> tuple[list[str], list[str]]:
    """Variables disponibles para orientar prevención antes del siniestro."""
    categorical = ["vias", "corregimiento"]
    numeric = [
        "anio",
        "mes",
        "dia_semana_num",
        "hora",
        "es_fin_de_semana",
        "mes_seno",
        "mes_coseno",
        "hora_seno",
        "hora_coseno",
    ]
    return categorical, numeric


def prepare_monthly_counts(incidents: pd.DataFrame) -> pd.DataFrame:
    """Crea un panel mensual por corregimiento, incluyendo meses sin siniestros.

    La unidad de decisión es el corregimiento-mes: permite a la Secretaría de
    Tránsito priorizar acciones mensuales. Los lags se calculan exclusivamente
    con meses anteriores, de modo que no incorporan información del futuro.
    """
    start = incidents["fecha"].min().to_period("M")
    end = incidents["fecha"].max().to_period("M")
    periods = pd.period_range(start, end, freq="M")
    locations = pd.DataFrame({"corregimiento": sorted(incidents["corregimiento"].unique())})
    panel = locations.merge(pd.DataFrame({"periodo": periods}), how="cross")
    panel["fecha"] = panel["periodo"].dt.to_timestamp()

    observed = (
        incidents.assign(periodo=incidents["fecha"].dt.to_period("M"))
        .groupby(["corregimiento", "periodo"], as_index=False)
        .size()
        .rename(columns={"size": TARGET_REGRESSION})
    )
    panel = panel.merge(observed, on=["corregimiento", "periodo"], how="left")
    panel[TARGET_REGRESSION] = panel[TARGET_REGRESSION].fillna(0).astype(int)
    panel = panel.sort_values(["corregimiento", "fecha"]).reset_index(drop=True)
    group = panel.groupby("corregimiento", observed=True)[TARGET_REGRESSION]
    panel["siniestros_lag_1"] = group.shift(1).fillna(0)
    panel["promedio_3_meses_previo"] = group.transform(
        lambda values: values.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0)
    panel["anio"] = panel["fecha"].dt.year.astype(int)
    panel["mes"] = panel["fecha"].dt.month.astype(int)
    panel["mes_seno"] = np.sin(2 * np.pi * panel["mes"] / 12)
    panel["mes_coseno"] = np.cos(2 * np.pi * panel["mes"] / 12)
    return panel


def regression_features() -> tuple[list[str], list[str]]:
    categorical = ["corregimiento"]
    numeric = [
        "anio",
        "mes",
        "mes_seno",
        "mes_coseno",
        "siniestros_lag_1",
        "promedio_3_meses_previo",
    ]
    return categorical, numeric


def time_split(data: pd.DataFrame, year_column: str = "anio") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mantiene el último año completo/observado como prueba temporal."""
    last_year = int(data[year_column].max())
    train = data.loc[data[year_column] < last_year].copy()
    test = data.loc[data[year_column] == last_year].copy()
    if train.empty or test.empty:
        raise ValueError("No hay suficiente cobertura temporal para una partición temporal.")
    return train, test
