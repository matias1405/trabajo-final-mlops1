from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mlflow
import requests
import os

def get_model():
    client = mlflow.MlflowClient(**context)
    model_name = "PredictionMovies"
    staging_model = client.get_model_version_by_alias(model_name,"staging")
    version = staging_model.version
    return version


def promote_model(**context):
    version = context["ti"].xcom_pull(task_ids="get_model")
    model_name = "PredictionMovies"
    alias = "production"
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(model_name, alias, version)
    version_tag = f"v{version}"
    client.set_model_version_tag(model_name,version,"release",version_tag)

    url = "http://fastapi-production:8000/model/reload"
    response = requests.post(
        url,
        headers={"X-Reload-Token": os.environ["MODEL_RELOAD_TOKEN"]},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


with DAG(
    dag_id="promote-model",
    description="Obtiene modelo candidate y lo promociona a PROD",
    schedule=None,
    catchup=False,
    tags=["mlops-tp", "prediction-movies"],
) as dag:
    t_get = PythonOperator(task_id="get_model", python_callable=get_model)
    t_promote = PythonOperator(task_id="promote_model", python_callable=promote_model)


    t_get >> t_promote