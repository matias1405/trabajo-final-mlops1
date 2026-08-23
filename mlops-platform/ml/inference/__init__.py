"""Inference helpers for the movie profitability model."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import mlflow
import numpy as np
import mlflow.sklearn
import pandas as pd


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


def _normalize_token(value: str) -> str:
    return str(value).strip().replace(" ", "_")


def _build_feature_frame(model, payload: dict[str, Any]) -> pd.DataFrame:
    feature_names = list(getattr(model, "feature_names_in_", []))
    row = {name: 0 for name in feature_names}

    budget = float(payload.get("budget", 0.0))
    runtime = float(payload.get("runtime", 0.0))
    original_language = _normalize_token(payload.get("original_language", "other"))
    genres = payload.get("genres") or []
    production_countries = payload.get("production_countries") or []
    production_companies = payload.get("production_companies") or []

    if "budget" in row:
        row["budget"] = float(np.log1p(budget))
    if "runtime" in row:
        row["runtime"] = runtime
    if "release_month" in row:
        row["release_month"] = int(payload.get("release_month") or datetime.utcnow().month)
    if "n_genres" in row:
        row["n_genres"] = len(genres)

    lang_candidates = [f"lang_{original_language}", "lang_other"]
    for column in lang_candidates:
        if column in row:
            row[column] = 1
            break

    for genre in genres:
        column = f"genre_{_normalize_token(genre)}"
        if column in row:
            row[column] = 1

    for country in production_countries:
        column = f"country_{_normalize_token(country)}"
        if column in row:
            row[column] = 1

    for company in production_companies:
        column = f"company_{_normalize_token(company)}"
        if column in row:
            row[column] = 1

    if not feature_names:
        feature_names = list(row.keys())

    return pd.DataFrame([[row[name] for name in feature_names]], columns=feature_names)


def predict(model, payload: dict[str, Any]) -> dict[str, Any]:
    features = _build_feature_frame(model, payload)
    prediction = model.predict(features)[0]

    response: dict[str, Any] = {
        "prediction": int(prediction),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        response["probability"] = float(proba[1])

    return response
