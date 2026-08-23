"""DAG de entrenamiento del modelo de rentabilidad de películas.

Orquesta el flujo descrito en el README: carga de datos -> validación ->
limpieza -> feature engineering -> entrenamiento -> evaluación -> registro
en MLflow. Cada tarea delega en el paquete ml/ (training, preprocessing).
"""
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
import json

import boto3
import joblib
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from botocore.exceptions import ClientError
import requests

import ml.v1.preprocessing as v1_preprocessing
import ml.v1.training as v1_training
import ml.v2.preprocessing as v2_preprocessing
import ml.v2.training as v2_training

RAW_DATA_KEY = "raw/TMDB_movie_dataset_v11.csv"
RAW_DATA_LOCAL_PATH = "/opt/airflow/data/raw/TMDB_movie_dataset_v11.csv"
WORKDIR = Path("/opt/airflow/data/processed/prediction_movies")
CLEAN_DATA_LOCAL_PATH = WORKDIR / "cleaned.csv"
X_TRAIN_LOCAL_PATH = WORKDIR / "X_train.csv"
X_TEST_LOCAL_PATH = WORKDIR / "X_test.csv"
Y_TRAIN_LOCAL_PATH = WORKDIR / "y_train.csv"
Y_TEST_LOCAL_PATH = WORKDIR / "y_test.csv"
MODEL_STAGING_LOCAL_PATH = WORKDIR / "model_v2.joblib"
METRICS_STAGING_LOCAL_PATH = WORKDIR / "metrics_v2.json"
MODEL_PRODUCTION_LOCAL_PATH = WORKDIR / "model_v1.joblib"
METRICS_PRODUCTION_LOCAL_PATH = WORKDIR / "metrics_v1.json"
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
    # lo sube desde un volumen montado. Se devuelve el path (no el DataFrame)
    # por XCom: el archivo pesa ~600MB y no es viable pasarlo por el backend
    # store de Airflow(Postgres).
    os.makedirs(os.path.dirname(RAW_DATA_LOCAL_PATH), exist_ok=True)
    bucket = os.environ["DATALAKE_BUCKET_NAME"]
    s3 = _build_s3_client()

    if not _raw_data_exists_in_minio(s3, bucket):
        s3.upload_file(LOCAL_DATASET_PATH , bucket, RAW_DATA_KEY)

    s3.download_file(bucket, RAW_DATA_KEY, RAW_DATA_LOCAL_PATH)


def validate_data(**context):
    df = pd.read_csv(RAW_DATA_LOCAL_PATH)
    v1_preprocessing.validate(df)


def clean_data(**context):
    df = pd.read_csv(RAW_DATA_LOCAL_PATH)
    cleaned_df = v1_preprocessing.clean(df)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_DATA_LOCAL_PATH, index=False)


def split_v1(**context):
    df = pd.read_csv(CLEAN_DATA_LOCAL_PATH)
    X_train, X_test, y_train, y_test = v1_preprocessing.split(df)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_LOCAL_PATH, index=False)
    X_test.to_csv(X_TEST_LOCAL_PATH, index=False)
    y_train.to_csv(Y_TRAIN_LOCAL_PATH, index=False, header=True)
    y_test.to_csv(Y_TEST_LOCAL_PATH, index=False, header=True)


def split_v2(**context):
    df = pd.read_csv(CLEAN_DATA_LOCAL_PATH)
    X_train, X_test, y_train, y_test = v1_preprocessing.split(df)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_LOCAL_PATH, index=False)
    X_test.to_csv(X_TEST_LOCAL_PATH, index=False)
    y_train.to_csv(Y_TRAIN_LOCAL_PATH, index=False, header=True)
    y_test.to_csv(Y_TEST_LOCAL_PATH, index=False, header=True)


def train_model_v1(**context):
    X_train = pd.read_csv(X_TRAIN_LOCAL_PATH)
    y_train = pd.read_csv(Y_TRAIN_LOCAL_PATH).squeeze("columns")
    model = v1_training.train(X_train, y_train)
    joblib.dump(model, MODEL_PRODUCTION_LOCAL_PATH)


def train_model_v2(**context):
    X_train = pd.read_csv(X_TRAIN_LOCAL_PATH)
    y_train = pd.read_csv(Y_TRAIN_LOCAL_PATH).squeeze("columns")
    model = v2_training.train(X_train, y_train)
    joblib.dump(model, MODEL_STAGING_LOCAL_PATH)


def evaluate_model_v1(**context):
    model = joblib.load(str(MODEL_PRODUCTION_LOCAL_PATH))
    X_test = pd.read_csv(X_TEST_LOCAL_PATH)
    y_test = pd.read_csv(Y_TEST_LOCAL_PATH).squeeze("columns")
    metrics = v1_training.evaluate(model, X_test, y_test)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PRODUCTION_LOCAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def evaluate_model_v2(**context):
    model = joblib.load(str(MODEL_STAGING_LOCAL_PATH))
    X_test = pd.read_csv(X_TEST_LOCAL_PATH)
    y_test = pd.read_csv(Y_TEST_LOCAL_PATH).squeeze("columns")
    metrics = v2_training.evaluate(model, X_test, y_test)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_STAGING_LOCAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)


def register_model_v1(**context):
    with open(METRICS_PRODUCTION_LOCAL_PATH, "r", encoding="utf-8") as fh:
        run_metrics = json.load(fh)
    model = joblib.load(str(MODEL_PRODUCTION_LOCAL_PATH))
    return v1_training.register_model(model, run_metrics)


def register_model_v2(**context):
    with open(METRICS_STAGING_LOCAL_PATH, "r", encoding="utf-8") as fh:
        run_metrics = json.load(fh)
    model = joblib.load(str(MODEL_STAGING_LOCAL_PATH))
    return v2_training.register_model(model, run_metrics)


def reload_model_v1(**context):
    api_url = "http://fastapi-production:8000/model/reload"
    reload_token = os.environ["MODEL_RELOAD_TOKEN"]
    response = requests.post(
        api_url,
        headers={
            "Authorization": f"Bearer {reload_token}",
        },
        timeout=30,
    )
    response.raise_for_status()
    print("Model reload requested successfully")
    print(response.json())


def reload_model_v2(**context):
    api_url = "http://fastapi-staging:8000/model/reload"
    reload_token = os.environ["MODEL_RELOAD_TOKEN"]
    response = requests.post(
        api_url,
        headers={
            "Authorization": f"Bearer {reload_token}",
        },
        timeout=30,
    )
    response.raise_for_status()
    print("Model reload requested successfully")
    print(response.json())


with DAG(
    dag_id="train-initial-models-prediction-movies",
    description="Entrena y registra los modelos iniciales de rentabilidad de películas v1 y v2-RC",
    schedule="@once",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops-tp", "prediction-movies"],
) as dag:

    # ==========================================
    # ETAPAS COMUNES
    # ==========================================

    t_load      = PythonOperator(task_id="load_data", python_callable=load_data)
    t_validate  = PythonOperator(task_id="validate_data",python_callable=validate_data)
    t_clean     = PythonOperator(task_id="clean_data",python_callable=clean_data)

    # ==========================================
    # MODELO V1
    # ==========================================

    t_split_v1      = PythonOperator(task_id="split_v1",python_callable=split_v1)
    t_train_v1      = PythonOperator(task_id="train_model_v1", python_callable=train_model_v1)
    t_evaluate_v1   = PythonOperator(task_id="evaluate_model_v1", python_callable=evaluate_model_v1)
    t_register_v1   = PythonOperator(task_id="register_model_v1", python_callable=register_model_v1)
    t_reload_v1     = PythonOperator(task_id="reload_model_v1", python_callable=reload_model_v1)

    # ==========================================
    # MODELO V2
    # ==========================================

    t_split_v2      = PythonOperator(task_id="split_v2",python_callable=split_v2)
    t_train_v2      = PythonOperator(task_id="train_model_v2", python_callable=train_model_v2)
    t_evaluate_v2   = PythonOperator(task_id="evaluate_model_v2", python_callable=evaluate_model_v2)
    t_register_v2   = PythonOperator(task_id="register_model_v2", python_callable=register_model_v2)
    t_reload_v2     = PythonOperator(task_id="reload_model_v2", python_callable=reload_model_v2)

    # ==========================================
    # DEPENDENCIAS
    # ==========================================

    (
        t_load
        >> t_validate
        >> t_clean
        >> [t_split_v1, t_split_v2]
    )

    t_split_v1 >> t_train_v1 >> t_evaluate_v1 >> t_register_v1 >> t_reload_v1
    t_split_v2 >> t_train_v2 >> t_evaluate_v2 >> t_register_v2 >> t_reload_v2

