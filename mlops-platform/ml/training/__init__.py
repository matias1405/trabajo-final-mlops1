"""Entrenamiento y evaluación del modelo de rentabilidad de películas, y
registro del resultado en el Model Registry de MLflow.
"""


def train(x_train, y_train):
    raise NotImplementedError


def evaluate(model, x_test, y_test):
    raise NotImplementedError


def register_model(model, run_metrics, model_name="PredictionMovies"):
    raise NotImplementedError
