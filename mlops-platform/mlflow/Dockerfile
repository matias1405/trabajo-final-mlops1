FROM python:3.12-slim

WORKDIR /mlflow

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

# Backend Store (Postgres) y Artifact Store (MinIO) se configuran vía
# variables de entorno inyectadas por docker-compose, no hardcodeadas aquí.
CMD mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --default-artifact-root "${MLFLOW_ARTIFACT_ROOT}"
