"""Entrenamiento y evaluación del modelo de rentabilidad de películas, y
registro del resultado en el Model Registry de MLflow.
"""

import os
import tempfile

import joblib
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from .models import create_random_forest, create_svm


def train(x_train, y_train, model_version="v1"):
    if model_version == "v1":
        model = create_random_forest()
    elif model_version == "v2-rc":
        model = create_svm()
    else:
        raise ValueError(f"Versión de modelo no soportada: {model_version}")

    model.fit(x_train, y_train)

    return model


def evaluate(model, x_test, y_test):
    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }


def register_model(model, run_metrics, encoder, model_name="PredictionMovies"):
    with mlflow.start_run():
        mlflow.log_metrics(run_metrics)
        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        # Se persiste junto con el modelo (mismo run) para que ml/inference
        # pueda reproducir el mismo encoding sin recalcular top-N ni
        # depender de model.feature_names_in_.
        with tempfile.TemporaryDirectory() as tmp_dir:
            encoder_path = os.path.join(tmp_dir, "feature_encoder.joblib")
            joblib.dump(encoder, encoder_path)
            mlflow.log_artifact(encoder_path, artifact_path="encoder")

    return model_info.registered_model_version


def promote_model(model_name, version, alias):
    """Asigna un alias de MLflow (ej. "staging", "production") a una versión
    ya registrada. Este es el paso que efectivamente hace que
    `models:/{model_name}@{alias}` resuelva a esa versión (registrar un
    modelo por sí solo no le asigna ningún alias).
    """
    client = MlflowClient()
    client.set_registered_model_alias(name=model_name, alias=alias, version=version)


def get_model_version_by_alias(model_name, alias):
    client = MlflowClient()
    return client.get_model_version_by_alias(model_name, alias).version
