"""API REST que expone el modelo de rentabilidad de películas.

Una misma imagen se usa para los ambientes Staging y Production: la variable
de entorno MODEL_STAGE determina qué versión del modelo (registrada en el
Model Registry de MLflow) se carga al iniciar el servicio. El modelo
permanece en memoria hasta que el proceso se reinicia.
"""
import os

import mlflow
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = os.environ.get("MODEL_NAME", "PredictionMovies")
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Staging")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title=f"PredictionMovies API ({MODEL_STAGE})")

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
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_stage": MODEL_STAGE, "model_loaded": model is not None}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict:
    # TODO: aplicar el mismo preprocesamiento/encoding usado en entrenamiento
    # (ver ml/preprocessing en mlops-platform) antes de llamar a model.predict.
    raise NotImplementedError
