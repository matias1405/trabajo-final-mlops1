"""Carga del modelo publicado en el Model Registry de MLflow (por stage) y
preprocesamiento de un request de predicción con el mismo pipeline de
features usado en entrenamiento. Consumido por client-app/fastapi.
"""


def load_model(model_name: str, stage: str):
    raise NotImplementedError


def predict(model, payload: dict):
    raise NotImplementedError
