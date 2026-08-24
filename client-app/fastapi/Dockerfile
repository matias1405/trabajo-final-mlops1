FROM python:3.12-slim@sha256:2c941e860699f878900b0edc2403613c234d4b32eda3cc9fa7036991a2a63c4a

WORKDIR /app

COPY ./client-app/fastapi/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY prediction_movies_ml/ml /tmp/prediction_movies_ml/ml
COPY prediction_movies_ml/pyproject.toml /tmp/prediction_movies_ml/pyproject.toml

RUN pip install --no-cache-dir /tmp/prediction_movies_ml/ \
    && rm -rf /tmp/prediction_movies_ml

COPY ./client-app/fastapi/app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
