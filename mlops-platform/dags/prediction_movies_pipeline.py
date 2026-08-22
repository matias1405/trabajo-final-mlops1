"""DAG de entrenamiento del modelo de rentabilidad de películas.

Orquesta el flujo descrito en el README: carga de datos -> validación ->
limpieza -> feature engineering -> entrenamiento -> evaluación -> registro
en MLflow. Cada tarea delega en el paquete ml/ (training, preprocessing).
"""
import os
import shutil
import tempfile
from datetime import datetime

import boto3
from airflow import DAG
from airflow.operators.python import PythonOperator
from botocore.exceptions import ClientError

from ml import preprocessing, training

RAW_DATA_KEY = "raw/TMDB_movie_dataset_v11.csv"
RAW_DATA_LOCAL_PATH = "/opt/airflow/data/raw/TMDB_movie_dataset_v11.csv"

# Dataset original en Kaggle: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies

LOCAL_DATASET_PATH = "/opt/airflow/dataset/TMDB_movie_dataset_v11.csv"

def _build_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    )


def _raw_data_exists_in_minio(s3, bucket):
    try:
        s3.head_object(Bucket=bucket, Key=RAW_DATA_KEY)
        return True
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "404":
            raise
        return False


def load_data(**_):
    # Descarga el CSV original desde el bucket Raw Data de MinIO (bucket
    # `datalake`, prefijo `raw/`) a un path local compartido por los
    # contenedores de Airflow. Si todavía no está en MinIO, primero lo
    # descarga de Kaggle y lo sube (automatiza el paso manual de subir el
    # dataset). Se devuelve el path (no el DataFrame) por XCom: el archivo
    # pesa ~600MB y no es viable pasarlo por el backend store de Airflow
    # (Postgres).
    os.makedirs(os.path.dirname(RAW_DATA_LOCAL_PATH), exist_ok=True)
    bucket = os.environ["DATALAKE_BUCKET_NAME"]
    s3 = _build_s3_client()

    if not _raw_data_exists_in_minio(s3, bucket):
        s3.upload_file(LOCAL_DATASET_PATH , bucket, RAW_DATA_KEY)

    s3.download_file(bucket, RAW_DATA_KEY, RAW_DATA_LOCAL_PATH)
    return RAW_DATA_LOCAL_PATH


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
