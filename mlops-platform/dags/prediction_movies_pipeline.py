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

from ml import preprocessing, training

RAW_DATA_KEY = "raw/TMDB_movie_dataset_v11.csv"
RAW_DATA_LOCAL_PATH = "/opt/airflow/data/raw/TMDB_movie_dataset_v11.csv"
WORKDIR = Path("/opt/airflow/data/processed/prediction_movies")
CLEAN_DATA_LOCAL_PATH = WORKDIR / "cleaned.csv"
X_TRAIN_LOCAL_PATH = WORKDIR / "X_train.csv"
X_TEST_LOCAL_PATH = WORKDIR / "X_test.csv"
Y_TRAIN_LOCAL_PATH = WORKDIR / "y_train.csv"
Y_TEST_LOCAL_PATH = WORKDIR / "y_test.csv"
MODEL_LOCAL_PATH = WORKDIR / "model.joblib"
METRICS_LOCAL_PATH = WORKDIR / "metrics.json"
ENCODER_LOCAL_PATH = WORKDIR / "encoder.joblib"
MODEL_NAME = os.environ.get("MODEL_NAME", "PredictionMovies")

# Dataset original en Kaggle: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
KAGGLE_DATASET = "asaniczka/tmdb-movies-dataset-2023-930k-movies"
KAGGLE_DATASET_FILENAME = "TMDB_movie_dataset_v11.csv"


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


def _download_dataset_from_kaggle(destination_path):
    # Requiere la credencial KAGGLE_API_TOKEN (ver .env.template). El paquete
    # `kaggle` se autentica solo al importarlo (usa KAGGLE_API_TOKEN del
    # entorno y expone el cliente ya autenticado como `kaggle.api`), así que
    # no hay que instanciar KaggleApi() ni llamar a .authenticate() a mano:
    # el import ya consume esa variable de entorno.
    import kaggle

    with tempfile.TemporaryDirectory() as tmp_dir:
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=tmp_dir, unzip=True)
        shutil.copy(os.path.join(tmp_dir, KAGGLE_DATASET_FILENAME), destination_path)


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
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_download_path = os.path.join(tmp_dir, KAGGLE_DATASET_FILENAME)
            _download_dataset_from_kaggle(local_download_path)
            s3.upload_file(local_download_path, bucket, RAW_DATA_KEY)

    s3.download_file(bucket, RAW_DATA_KEY, RAW_DATA_LOCAL_PATH)
    return RAW_DATA_LOCAL_PATH


def validate_data(**context):
    raw_data_path = context["ti"].xcom_pull(task_ids="load_data")
    df = pd.read_csv(raw_data_path)
    preprocessing.validate(df)
    return raw_data_path


def clean_data(**context):
    raw_data_path = context["ti"].xcom_pull(task_ids="validate_data")
    df = pd.read_csv(raw_data_path)
    cleaned_df = preprocessing.clean(df)
    WORKDIR.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(CLEAN_DATA_LOCAL_PATH, index=False)
    return str(CLEAN_DATA_LOCAL_PATH)


def feature_engineering(**context):
    cleaned_data_path = context["ti"].xcom_pull(task_ids="clean_data")
    df = pd.read_csv(cleaned_data_path)
    X_train, X_test, y_train, y_test, encoder = preprocessing.engineer_features(df)

    WORKDIR.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_LOCAL_PATH, index=False)
    X_test.to_csv(X_TEST_LOCAL_PATH, index=False)
    y_train.to_csv(Y_TRAIN_LOCAL_PATH, index=False, header=True)
    y_test.to_csv(Y_TEST_LOCAL_PATH, index=False, header=True)
    joblib.dump(encoder, ENCODER_LOCAL_PATH)

    return str(WORKDIR)


def train_model(**context):
    feature_dir = Path(context["ti"].xcom_pull(task_ids="feature_engineering"))
    X_train = pd.read_csv(feature_dir / "X_train.csv")
    y_train = pd.read_csv(feature_dir / "y_train.csv").squeeze("columns")

    model = training.train(X_train, y_train)
    joblib.dump(model, MODEL_LOCAL_PATH)
    return str(MODEL_LOCAL_PATH)


def evaluate_model(**context):
    feature_dir = Path(context["ti"].xcom_pull(task_ids="feature_engineering"))
    model_path = context["ti"].xcom_pull(task_ids="train_model")

    model = joblib.load(model_path)
    X_test = pd.read_csv(feature_dir / "X_test.csv")
    y_test = pd.read_csv(feature_dir / "y_test.csv").squeeze("columns")
    metrics = training.evaluate(model, X_test, y_test)

    WORKDIR.mkdir(parents=True, exist_ok=True)
    with open(METRICS_LOCAL_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    return str(METRICS_LOCAL_PATH)


def register_model(**context):
    model_path = context["ti"].xcom_pull(task_ids="train_model")
    metrics_path = context["ti"].xcom_pull(task_ids="evaluate_model")
    feature_dir = Path(context["ti"].xcom_pull(task_ids="feature_engineering"))

    with open(metrics_path, "r", encoding="utf-8") as fh:
        run_metrics = json.load(fh)

    model = joblib.load(model_path)
    encoder = joblib.load(feature_dir / "encoder.joblib")
    return training.register_model(model, run_metrics, encoder, model_name=MODEL_NAME)


def promote_to_staging(**context):
    # Toda versión recién entrenada entra al Registry sin alias asignado
    # (register_model no la promueve a ningún lado): se aterriza en Staging
    # para que el equipo de QA la valide desde el frontend correspondiente
    # antes de que alguien la promueva a Production (ver dags/promote_model.py).
    version = context["ti"].xcom_pull(task_ids="register_model")
    training.promote_model(MODEL_NAME, version, "staging")


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
    t_promote_staging = PythonOperator(task_id="promote_to_staging", python_callable=promote_to_staging)

    t_load >> t_validate >> t_clean >> t_features >> t_train >> t_evaluate >> t_register >> t_promote_staging
