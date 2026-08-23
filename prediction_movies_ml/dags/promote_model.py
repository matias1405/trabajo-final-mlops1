from airflow import DAG
from airflow.operators.python import PythonOperator

def get_model():
    print("hello world")

def promote_model():
    print("hello world")

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