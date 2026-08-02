"""DAG de entrenamiento del modelo de rentabilidad de películas.

Orquesta el flujo descrito en el README: carga de datos -> validación ->
limpieza -> feature engineering -> entrenamiento -> evaluación -> registro
en MLflow. Cada tarea delega en el paquete ml/ (training, preprocessing).
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml import preprocessing, training


def load_data(**_):
    # TODO: leer TMDB_movie_dataset_v11.csv desde el bucket Raw Data de MinIO.
    raise NotImplementedError


def validate_data(**_):
    return preprocessing.validate(None)


def clean_data(**_):
    return preprocessing.clean(None)


def feature_engineering(**_):
    return preprocessing.engineer_features(None)


def train_model(**_):
    return training.train(None, None)


def evaluate_model(**_):
    return training.evaluate(None, None, None)


def register_model(**_):
    return training.register_model(None, None)


with DAG(
    dag_id="prediction_movies_pipeline",
    description="Entrena y registra el modelo de rentabilidad de películas",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops-tp", "prediction-movies"],
) as dag:
    t_load = PythonOperator(task_id="load_data", python_callable=load_data)
    t_validate = PythonOperator(task_id="validate_data", python_callable=validate_data)
    t_clean = PythonOperator(task_id="clean_data", python_callable=clean_data)
    t_features = PythonOperator(task_id="feature_engineering", python_callable=feature_engineering)
    t_train = PythonOperator(task_id="train_model", python_callable=train_model)
    t_evaluate = PythonOperator(task_id="evaluate_model", python_callable=evaluate_model)
    t_register = PythonOperator(task_id="register_model", python_callable=register_model)

    t_load >> t_validate >> t_clean >> t_features >> t_train >> t_evaluate >> t_register
