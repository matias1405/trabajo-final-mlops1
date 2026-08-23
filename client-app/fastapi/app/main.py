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
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title=f"PredictionMovies API ({MODEL_ALIAS})")

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
            alias=MODEL_ALIAS
        )
    except Exception:
        model = None


@app.get("/health")
def health(response):
    if model is None:
        response.status_code = 503

        return {
            "status": "waiting_for_model",
            "model_ref": MODEL_ALIAS,
            "model_loaded": False
        }

    return {
        "status": "ok",
        "model_ref": MODEL_ALIAS,
        "model_loaded": True
    }


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return predict_model(model, request.model_dump())


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
        alias=MODEL_ALIAS
    )

    return {
        "status": "ok",
        "model_ref": MODEL_ALIAS,
    }


def load_model(
    model_name: str,
    *,
    alias: str | None = None
):

    model_uri = f"models:/{model_name}@{alias}"

    return mlflow.sklearn.load_model(model_uri)


def predict_model(model, payload: dict) -> dict:
    df = pd.DataFrame([payload])

    prediction = model.predict(df)[0]

    response: dict = {
        "prediction": int(prediction),
    }

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(df)[0]
        response["probability"] = float(proba[1])

    return response