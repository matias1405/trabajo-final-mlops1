FROM apache/airflow:slim-2.10.4-python3.12@sha256:696ce48a24c05033f597b8df7a41cad82cdf2d6576809022a3d0734eb8d1f3b9

# Dependencias del pipeline de entrenamiento (pandas, scikit-learn, catboost,
# lightgbm, cliente de MLflow) además de las que trae la imagen base de Airflow.
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt