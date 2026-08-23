"""Inference helpers for the movie profitability model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import joblib
import mlflow
import mlflow.artifacts
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient


def load_model(
    model_name: str,
    *,
    alias: str | None = None,
    stage: str | None = None,
    tracking_uri: str | None = None,
):
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    if alias:
        model_uri = f"models:/{model_name}@{alias}"
    elif stage:
        model_uri = f"models:/{model_name}/{stage}"
    else:
        model_uri = f"models:/{model_name}/Staging"

    return mlflow.sklearn.load_model(model_uri)


def _resolve_model_version(model_name: str, *, alias: str | None, stage: str | None):
    client = MlflowClient()
    if alias:
        return client.get_model_version_by_alias(model_name, alias)
    if stage:
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            raise ValueError(f"No hay ninguna versión de {model_name} en stage {stage}")
        return versions[0]
    return client.get_model_version_by_alias(model_name, "staging")


def load_encoder(
    model_name: str,
    *,
    alias: str | None = None,
    stage: str | None = None,
    tracking_uri: str | None = None,
):
    """Carga el FeatureEncoder (ml.preprocessing.FeatureEncoder) persistido
    junto con el modelo por ml.training.register_model, para reproducir
    exactamente el mismo encoding usado en entrenamiento.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    model_version = _resolve_model_version(model_name, alias=alias, stage=stage)
    local_path = mlflow.artifacts.download_artifacts(
        run_id=model_version.run_id,
        artifact_path="encoder/feature_encoder.joblib",
    )
    return joblib.load(local_path)


def predict(model, encoder, payload: dict[str, Any]) -> dict[str, Any]:
    row = {
        "budget": float(payload.get("budget", 0.0)),
        "runtime": float(payload.get("runtime", 0.0)),
        "original_language": payload.get("original_language", "other"),
        "genres": payload.get("genres") or [],
        "production_countries": payload.get("production_countries") or [],
        "production_companies": payload.get("production_companies") or [],
        "release_month": int(payload.get("release_month") or datetime.utcnow().month),
    }

    features = encoder.transform(pd.DataFrame([row]))
    prediction = model.predict(features)[0]

    response: dict[str, Any] = {"prediction": int(prediction)}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        response["probability"] = float(proba[1])

    return response
