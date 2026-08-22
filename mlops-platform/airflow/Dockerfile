FROM apache/airflow:slim-2.10.4-python3.12@sha256:696ce48a24c05033f597b8df7a41cad82cdf2d6576809022a3d0734eb8d1f3b9

# Dependencias solo del providers para conexion a la base postgres y minio
RUN pip install --no-cache-dir apache-airflow-providers-amazon==9.1.0 apache-airflow-providers-postgres==5.14.0
