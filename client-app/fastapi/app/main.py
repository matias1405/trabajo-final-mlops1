"""API REST que expone el modelo de rentabilidad de películas.

Una misma imagen se usa para los ambientes Staging y Production: la variable
de entorno MODEL_ALIAS define el alias de MLflow que se carga al iniciar el
servicio. Como respaldo se mantiene MODEL_STAGE para compatibilidad con la
vieja API de stages.
"""
import os

import mlflow
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml import inference

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
def load_model() -> None:
    global model
    model = inference.load_model(
        MODEL_NAME,
        alias=MODEL_ALIAS,
        stage=MODEL_STAGE,
        tracking_uri=MLFLOW_TRACKING_URI,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_ref": MODEL_REF, "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    return inference.predict(model, request.model_dump())
