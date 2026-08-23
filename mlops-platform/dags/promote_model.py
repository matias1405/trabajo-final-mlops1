"""DAG para promover a Production el modelo actualmente en Staging.

Se dispara manualmente una vez que el equipo de QA validó el modelo desde el
Frontend/FastAPI de Staging (ver README, sección "Promoción de modelos").
No hace nada automático: quien lo dispara es responsable de haber validado
el modelo antes.
"""
import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator

from ml import training

MODEL_NAME = os.environ.get("MODEL_NAME", "PredictionMovies")


def promote_to_production(**_):
    version = training.get_model_version_by_alias(MODEL_NAME, "staging")
    training.promote_model(MODEL_NAME, version, "production")
    print(f"{MODEL_NAME} version {version} promovido a production")


with DAG(
    dag_id="promote_model_to_production",
    description="Promueve a Production la versión de PredictionMovies actualmente en Staging",
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["mlops-tp", "prediction-movies"],
) as dag:
    t_promote = PythonOperator(task_id="promote_to_production", python_callable=promote_to_production)
