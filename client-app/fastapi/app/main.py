"""API REST que expone el modelo de rentabilidad de películas.

Una misma imagen se usa para los ambientes Staging y Production: la variable
de entorno MODEL_ALIAS define el alias de MLflow que se carga al iniciar el
servicio. Como respaldo se mantiene MODEL_STAGE para compatibilidad con la
vieja API de stages.
"""
import os

import mlflow
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

MODEL_NAME = os.environ.get("MODEL_NAME", "PredictionMovies")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS")
MODEL_STAGE = os.environ.get("MODEL_STAGE")
MODEL_REF = MODEL_ALIAS or MODEL_STAGE or "Staging"
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title=f"PredictionMovies API ({MODEL_REF})")

allowed_origins = [
    "http://localhost:3001",
    "http://localhost:3002",
    "http://127.0.0.1:3001",
    "http://127.0.0.1:3002",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


class PredictionRequest(BaseModel):
    budget: float
    runtime: float
    original_language: str
    genres: list[str]
    production_countries: list[str]
    production_companies: list[str]


@app.on_event("startup")
def load_model():
    global model
    try:
        model = load_model(
            MODEL_NAME,
            alias=MODEL_ALIAS,
            stage=MODEL_STAGE,
            tracking_uri=MLFLOW_TRACKING_URI,
        )
    except Exception:
        model = None


@app.on_event("startup")
def load_model() -> None:
    global model
    model = load_model(
        MODEL_NAME,
        alias=MODEL_ALIAS,
        stage=MODEL_STAGE,
        tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.get("/health")
def health(response: Response) -> dict:
    if model is None:
        response.status_code = 503

        return {
            "status": "waiting_for_model",
            "model_ref": MODEL_REF,
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_ref": MODEL_REF,
        "model_loaded": True
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predict(model, request.model_dump())


@app.post("/model/reload")
def reload_model(
    authorization: str = Header(...)
) -> dict:


    expected_token = os.environ["MODEL_RELOAD_TOKEN"]
    if authorization != f"Bearer {expected_token}":
        raise HTTPException(
            status_code=401,
            detail="Invalid token",
        )
    
    global model

    model = load_model(
        MODEL_NAME,
        alias=MODEL_ALIAS,
        stage=MODEL_STAGE,
        tracking_uri=MLFLOW_TRACKING_URI,
    )

    return {
        "status": "ok",
        "model_ref": MODEL_REF,
    }


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