FROM apache/airflow:2.10.4-python3.12

# Dependencias del pipeline de entrenamiento (pandas, scikit-learn, catboost,
# lightgbm, cliente de MLflow) además de las que trae la imagen base de Airflow.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
