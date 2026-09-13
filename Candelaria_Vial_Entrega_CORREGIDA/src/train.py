"""Entrenamiento, comparación y persistencia de modelos reproducibles."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from src.data import DATASET_ID, SOURCE_PAGE, load_raw_data
from src.features import (
    TARGET_CLASSIFICATION,
    TARGET_REGRESSION,
    classification_features,
    prepare_incidents,
    prepare_monthly_counts,
    regression_features,
    time_split,
)

ARTIFACT_DIR = Path("artifacts")
RANDOM_STATE = 42


def build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            ),
        ]
    )


def regression_models() -> dict[str, object]:
    return {
        "Baseline: promedio histórico": DummyRegressor(strategy="mean"),
        "Regresión Ridge": Ridge(alpha=8.0),
        "Árbol de decisión": DecisionTreeRegressor(
            max_depth=6, min_samples_leaf=8, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=4,
            max_features=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def classification_models() -> dict[str, object]:
    return {
        "Baseline: clase mayoritaria": DummyClassifier(strategy="prior"),
        "Regresión logística": LogisticRegression(
            C=0.5,
            class_weight="balanced",
            max_iter=2_000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        ),
        "Árbol de decisión": DecisionTreeClassifier(
            max_depth=6,
            min_samples_leaf=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350,
            max_depth=12,
            min_samples_leaf=6,
            max_features=0.8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def tune_regression_model(
    train_tune: pd.DataFrame,
    val_tune: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    estimator: object,
    param_distributions: dict,
) -> Pipeline:
    """Optimiza con validación temporal: train_tune -> val_tune."""
    features = categorical + numeric
    combined = pd.concat([train_tune, val_tune], ignore_index=True)
    test_fold = np.array([-1] * len(train_tune) + [0] * len(val_tune))
    splitter = PredefinedSplit(test_fold=test_fold)
    pipe = Pipeline([("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_distributions, n_iter=10,
        scoring="neg_mean_absolute_error", cv=splitter,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )
    search.fit(combined[features], combined[TARGET_REGRESSION])
    return search.best_estimator_


def tune_classification_model(
    train_tune: pd.DataFrame,
    val_tune: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    estimator: object,
    param_distributions: dict,
) -> Pipeline:
    """Optimiza con validación temporal: train_tune -> val_tune."""
    features = categorical + numeric
    combined = pd.concat([train_tune, val_tune], ignore_index=True)
    test_fold = np.array([-1] * len(train_tune) + [0] * len(val_tune))
    splitter = PredefinedSplit(test_fold=test_fold)
    pipe = Pipeline([("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)])
    search = RandomizedSearchCV(
        pipe, param_distributions=param_distributions, n_iter=10,
        scoring="roc_auc", cv=splitter,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True
    )
    search.fit(combined[features], combined[TARGET_CLASSIFICATION])
    return search.best_estimator_


def tune_regression_model(train_tune, val_tune, categorical, numeric, estimator, param_distributions):
    features = categorical + numeric
    combined = pd.concat([train_tune, val_tune], ignore_index=True)
    splitter = PredefinedSplit(test_fold=np.array([-1] * len(train_tune) + [0] * len(val_tune)))
    pipe = Pipeline([("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)])
    search = RandomizedSearchCV(pipe, param_distributions=param_distributions, n_iter=10,
                                scoring="neg_mean_absolute_error", cv=splitter,
                                random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    search.fit(combined[features], combined[TARGET_REGRESSION])
    return search.best_estimator_


def tune_classification_model(train_tune, val_tune, categorical, numeric, estimator, param_distributions):
    features = categorical + numeric
    combined = pd.concat([train_tune, val_tune], ignore_index=True)
    splitter = PredefinedSplit(test_fold=np.array([-1] * len(train_tune) + [0] * len(val_tune)))
    pipe = Pipeline([("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)])
    search = RandomizedSearchCV(pipe, param_distributions=param_distributions, n_iter=10,
                                scoring="roc_auc", cv=splitter,
                                random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    search.fit(combined[features], combined[TARGET_CLASSIFICATION])
    return search.best_estimator_

def regression_scores(actual: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(mean_squared_error(actual, prediction) ** 0.5),
        "R2": float(r2_score(actual, prediction)),
    }


def classification_scores(actual: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prediction = (probability >= 0.5).astype(int)
    return {
        "Accuracy": float(accuracy_score(actual, prediction)),
        "Precision": float(precision_score(actual, prediction, zero_division=0)),
        "Recall": float(recall_score(actual, prediction, zero_division=0)),
        "F1": float(f1_score(actual, prediction, zero_division=0)),
        "AUC_ROC": float(roc_auc_score(actual, probability)),
        "AUC_PR": float(average_precision_score(actual, probability)),
    }


def fit_problem(
    train: pd.DataFrame,
    test: pd.DataFrame,
    categorical: list[str],
    numeric: list[str],
    target: str,
    models: dict[str, object],
    problem: str,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    features = categorical + numeric
    fitted: dict[str, Pipeline] = {}
    results: list[dict[str, float | str]] = []
    for name, estimator in models.items():
        pipeline = Pipeline(
            [("preprocessor", build_preprocessor(categorical, numeric)), ("model", estimator)]
        )
        pipeline.fit(train[features], train[target])
        if problem == "regression":
            scores = regression_scores(test[target], pipeline.predict(test[features]))
        else:
            scores = classification_scores(test[target], pipeline.predict_proba(test[features])[:, 1])
        results.append({"modelo": name, **scores})
        fitted[name] = pipeline
    metric = "MAE" if problem == "regression" else "AUC_ROC"
    return fitted, pd.DataFrame(results).sort_values(
        metric, ascending=problem == "regression"
    )


def feature_importance(model: Pipeline, limit: int = 5) -> list[dict[str, float | str]]:
    """Obtiene una explicación global simple para el modelo seleccionado."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return []
    names = model.named_steps["preprocessor"].get_feature_names_out()
    values = estimator.feature_importances_
    order = np.argsort(values)[::-1][:limit]
    return [
        {"variable": str(names[index]), "importancia": float(values[index])}
        for index in order
    ]


def main() -> None:
    raw = load_raw_data()
    incidents = prepare_incidents(raw)
    monthly = prepare_monthly_counts(incidents)
    incidents_train, incidents_test = time_split(incidents)
    monthly_train, monthly_test = time_split(monthly)

    reg_categorical, reg_numeric = regression_features()
    clf_categorical, clf_numeric = classification_features()
    fitted_reg, regression_results = fit_problem(
        monthly_train,
        monthly_test,
        reg_categorical,
        reg_numeric,
        TARGET_REGRESSION,
        regression_models(),
        "regression",
    )
    fitted_clf, classification_results = fit_problem(
        incidents_train,
        incidents_test,
        clf_categorical,
        clf_numeric,
        TARGET_CLASSIFICATION,
        classification_models(),
        "classification",
    )

    # Optimización temporal: el año previo al test se usa como validación; el test queda intacto.
    test_year = int(incidents["anio"].max())
    validation_year = test_year - 1
    reg_tune_train = monthly[monthly["anio"] < validation_year].copy()
    reg_tune_val = monthly[monthly["anio"] == validation_year].copy()
    clf_tune_train = incidents[incidents["anio"] < validation_year].copy()
    clf_tune_val = incidents[incidents["anio"] == validation_year].copy()

    reg_spaces = {
        "Regresión Ridge": {"model__alpha": np.logspace(-2, 2, 20)},
        "Árbol de decisión": {"model__max_depth": [3, 4, 5, 6, 8, 10, None], "model__min_samples_leaf": [2, 4, 6, 8, 12, 16]},
        "Random Forest": {"model__n_estimators": [200, 300, 500], "model__max_depth": [5, 8, 10, 12, None], "model__min_samples_leaf": [2, 4, 6, 8], "model__max_features": [0.5, 0.8, 1.0]},
    }
    tuned_reg = {}
    for name in reg_spaces:
        tuned_reg[name] = tune_regression_model(reg_tune_train, reg_tune_val, reg_categorical, reg_numeric, regression_models()[name], reg_spaces[name])

    clf_spaces = {
        "Regresión logística": {"model__C": np.logspace(-2, 1, 15)},
        "Árbol de decisión": {"model__max_depth": [3, 4, 5, 6, 8, 10], "model__min_samples_leaf": [4, 8, 12, 16]},
        "Random Forest": {"model__n_estimators": [200, 350, 500], "model__max_depth": [6, 8, 12, 16, None], "model__min_samples_leaf": [2, 4, 6, 8], "model__max_features": [0.5, 0.8, 1.0]},
    }
    tuned_clf = {}
    for name in clf_spaces:
        tuned_clf[name] = tune_classification_model(clf_tune_train, clf_tune_val, clf_categorical, clf_numeric, classification_models()[name], clf_spaces[name])

    # Reentrenar los mejores parámetros con todo el periodo de entrenamiento y evaluar solo una vez en test.
    tuned_reg_rows = []
    for name, model in tuned_reg.items():
        model.fit(monthly_train[reg_categorical + reg_numeric], monthly_train[TARGET_REGRESSION])
        pred = model.predict(monthly_test[reg_categorical + reg_numeric])
        tuned_reg_rows.append({"modelo": name, **regression_scores(monthly_test[TARGET_REGRESSION], pred)})
    tuned_reg_results = pd.DataFrame(tuned_reg_rows).sort_values("MAE")

    tuned_clf_rows = []
    for name, model in tuned_clf.items():
        model.fit(incidents_train[clf_categorical + clf_numeric], incidents_train[TARGET_CLASSIFICATION])
        proba = model.predict_proba(incidents_test[clf_categorical + clf_numeric])[:, 1]
        tuned_clf_rows.append({"modelo": name, **classification_scores(incidents_test[TARGET_CLASSIFICATION], proba)})
    tuned_clf_results = pd.DataFrame(tuned_clf_rows).sort_values("AUC_ROC", ascending=False)

    fitted_reg.update(tuned_reg)
    fitted_clf.update(tuned_clf)
    regression_results = pd.concat([regression_results, tuned_reg_results.assign(modelo=lambda x: x["modelo"] + " (optimizado)")], ignore_index=True).sort_values("MAE")
    classification_results = pd.concat([classification_results, tuned_clf_results.assign(modelo=lambda x: x["modelo"] + " (optimizado)")], ignore_index=True).sort_values("AUC_ROC", ascending=False)

    # Optimización temporal: el año previo al test se usa como validación; el test queda intacto.
    test_year = int(incidents["anio"].max())
    validation_year = test_year - 1
    reg_tune_train = monthly[monthly["anio"] < validation_year].copy()
    reg_tune_val = monthly[monthly["anio"] == validation_year].copy()
    clf_tune_train = incidents[incidents["anio"] < validation_year].copy()
    clf_tune_val = incidents[incidents["anio"] == validation_year].copy()

    reg_spaces = {
        "Regresión Ridge": {"model__alpha": np.logspace(-2, 2, 20)},
        "Árbol de decisión": {"model__max_depth": [3, 4, 5, 6, 8, 10, None], "model__min_samples_leaf": [2, 4, 6, 8, 12, 16]},
        "Random Forest": {"model__n_estimators": [200, 300, 500], "model__max_depth": [5, 8, 10, 12, None], "model__min_samples_leaf": [2, 4, 6, 8], "model__max_features": [0.5, 0.8, 1.0]},
    }
    tuned_reg = {name: tune_regression_model(reg_tune_train, reg_tune_val, reg_categorical, reg_numeric, regression_models()[name], space) for name, space in reg_spaces.items()}

    clf_spaces = {
        "Regresión logística": {"model__C": np.logspace(-2, 1, 15)},
        "Árbol de decisión": {"model__max_depth": [3, 4, 5, 6, 8, 10], "model__min_samples_leaf": [4, 8, 12, 16]},
        "Random Forest": {"model__n_estimators": [200, 350, 500], "model__max_depth": [6, 8, 12, 16, None], "model__min_samples_leaf": [2, 4, 6, 8], "model__max_features": [0.5, 0.8, 1.0]},
    }
    tuned_clf = {name: tune_classification_model(clf_tune_train, clf_tune_val, clf_categorical, clf_numeric, classification_models()[name], space) for name, space in clf_spaces.items()}

    tuned_reg_rows = []
    for name, model in tuned_reg.items():
        model.fit(monthly_train[reg_categorical + reg_numeric], monthly_train[TARGET_REGRESSION])
        pred = model.predict(monthly_test[reg_categorical + reg_numeric])
        tuned_reg_rows.append({"modelo": name + " (optimizado)", **regression_scores(monthly_test[TARGET_REGRESSION], pred)})
    tuned_reg_results = pd.DataFrame(tuned_reg_rows)

    tuned_clf_rows = []
    for name, model in tuned_clf.items():
        model.fit(incidents_train[clf_categorical + clf_numeric], incidents_train[TARGET_CLASSIFICATION])
        proba = model.predict_proba(incidents_test[clf_categorical + clf_numeric])[:, 1]
        tuned_clf_rows.append({"modelo": name + " (optimizado)", **classification_scores(incidents_test[TARGET_CLASSIFICATION], proba)})
    tuned_clf_results = pd.DataFrame(tuned_clf_rows)

    regression_results = pd.concat([regression_results, tuned_reg_results], ignore_index=True).sort_values("MAE")
    classification_results = pd.concat([classification_results, tuned_clf_results], ignore_index=True).sort_values("AUC_ROC", ascending=False)
    fitted_reg.update({name + " (optimizado)": model for name, model in tuned_reg.items()})
    fitted_clf.update({name + " (optimizado)": model for name, model in tuned_clf.items()})

    ARTIFACT_DIR.mkdir(exist_ok=True)
    best_regression_name = regression_results.iloc[0]["modelo"]
    best_classification_name = classification_results.iloc[0]["modelo"]
    joblib.dump(fitted_reg[best_regression_name], ARTIFACT_DIR / "modelo_priorizacion_mensual.joblib")
    joblib.dump(fitted_clf[best_classification_name], ARTIFACT_DIR / "modelo_severidad.joblib")

    metadata = {
        "dataset_id": DATASET_ID,
        "source": SOURCE_PAGE,
        "registros_incidentes": int(len(incidents)),
        "registros_panel_mensual": int(len(monthly)),
        "fecha_minima": str(incidents["fecha"].min().date()),
        "fecha_maxima": str(incidents["fecha"].max().date()),
        "anio_entrenamiento_maximo": int(incidents_train["anio"].max()),
        "anio_prueba": int(incidents_test["anio"].min()),
        "modelo_regresion_seleccionado": best_regression_name,
        "modelo_clasificacion_seleccionado": best_classification_name,
        "clasificacion_apta_para_despliegue": bool(
            classification_results.iloc[0]["AUC_ROC"] >= 0.70
        ),
        "factores_importantes_regresion": feature_importance(
            fitted_reg[best_regression_name]
        ),
        "resultados_regresion": regression_results.round(4).to_dict(orient="records"),
        "resultados_clasificacion": classification_results.round(4).to_dict(orient="records"),
        "corregimientos": sorted(monthly["corregimiento"].unique().tolist()),
        "vias": sorted(incidents["vias"].unique().tolist()),
    }
    (ARTIFACT_DIR / "metricas.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Regresión — demanda mensual de siniestros")
    print(regression_results.round(4).to_string(index=False))
    print("\nClasificación — lesión o muerte")
    print(classification_results.round(4).to_string(index=False))
    print("\nModelos y métricas guardados en artifacts/.")


if __name__ == "__main__":
    main()
