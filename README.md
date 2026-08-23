# ML Models and Something More Inc. - MLOps Platform

Este proyecto implementa una plataforma de **MLOps** utilizando una arquitectura completamente desplegada mediante Docker Compose.

El objetivo es simular una plataforma productiva donde un equipo de Ciencia de Datos puede entrenar, versionar y publicar modelos de Machine Learning, mientras que distintas aplicaciones consumidoras pueden realizar inferencias mediante una API REST sin conocer los detalles internos del ciclo de vida del modelo.

La solución utiliza herramientas ampliamente utilizadas en la industria:

* Apache Airflow para la orquestación de pipelines de DataOps y MLOps.
* MLflow para el seguimiento de experimentos y el registro de modelos.
* MinIO para simulación de un servicio de almacenamiento S3 de AWS.
* PostgreSQL como base de datos para metadatos de MLflow.
* FastAPI para exponer los modelos mediante una REST API.
* React para el Frontend de la app cliente.
* Docker Compose para orquestar toda la plataforma.

---

# Integrantes:

- Matías Guillermo Alfaro
- Gonzalo Cuervo
- Nicolas Alberto Tonnelier
- Marina Andrea Racciatti

# Requisitos

Todo corre dentro de contenedores: no hace falta instalar Python, Node, Airflow, MLflow ni ninguna librería del proyecto en la máquina host. Lo único necesario es:

* **Docker Engine** con el plugin **Compose V2** (comando `docker compose`, no el binario standalone `docker-compose` v1 — el `docker-compose.yml` usa `depends_on: condition: service_completed_successfully`, que la v1 no soporta).
* **[Git LFS](https://git-lfs.com/)** instalado (`git lfs install`, una única vez por máquina) — `TMDB_movie_dataset_v11.csv` está versionado en el repo con LFS (ver "Dataset" más abajo); sin esto, al clonar sólo se obtiene un puntero de texto en vez del CSV real.
* **~5 GB de espacio libre en disco**: imágenes de los servicios + el dataset (`TMDB_movie_dataset_v11.csv`, ~600 MB) + los datos procesados que Airflow genera durante el entrenamiento.
* Los puertos por defecto libres en el host: `9000`/`9001` (MinIO), `5050` (MLflow), `8080` (Airflow), `8001`/`8002` (FastAPI Staging/Production), `3001`/`3002` (Frontend Staging/Production). Todos son configurables desde `.env` si alguno está ocupado (ver el aviso sobre macOS y el puerto 5000 más abajo).

---

# Puesta en marcha

1. **Configurar variables de entorno**

   ```bash
   cp .env.template .env
   ```

   Completar en el `.env` recién creado `AIRFLOW_FERNET_KEY` y `AIRFLOW_SECRET_KEY`, generándolos con los comandos que indica el propio `.env.template`.

   El resto de las variables (MinIO, Postgres, usuario admin de Airflow, puertos) ya vienen con valores demo utilizables tal cual para desarrollo local.

2. **Levantar servicios**

   ```bash
   docker compose up -d --build
   ```

   Y verificarlos:

   ```bash
   docker compose ps
   ```

   `postgres`, `mlflow` y `s3` (MinIO) deben estar `healthy`. `fastapi-staging` y `fastapi-production`, por otro lado, pueden fallar al arrancar, reiniciar y fallar en loop, porque todavía no existe ningún modelo `PredictionMovies` registrado. Pero se resuelve en el paso 4.

3. **Entrenar y registrar el primer modelo**

   Entrar a la UI de Airflow en `http://localhost:${AIRFLOW_PORT}` (usuario/contraseña: `AIRFLOW_ADMIN_USER`/`AIRFLOW_ADMIN_PASSWORD` del `.env`), despausar el DAG `prediction_movies_pipeline` y correrlo. 
   
   ![Entrenar modelo](./dag-entrenar-modelo.jpg)

   Equivalente por CLI:

   ```bash
   docker compose exec airflow-webserver airflow dags unpause prediction_movies_pipeline
   docker compose exec airflow-webserver airflow dags trigger prediction_movies_pipeline
   ```

   Esto sube el dataset en el repo (`mlops-platform/dataset/`) al bucket Raw Data de MinIO si todavía no está ahí (o lo toma de MinIO directamente si ya fue subido en una corrida anterior), lo valida, limpia, genera las features, entrena, evalúa, registra el modelo en el Model Registry de MLflow y lo promueve al alias `staging`.

   ![Promover staging](./dag-promote-staging.jpeg)

4. **Cargar el modelo en Staging**

   Una vez que el DAG termina en `success` (todo verde en la UI de Airflow o `http://localhost:${MLFLOW_PORT}`), reiniciar FastAPI Staging para que cargue el modelo recién promovido:

   ```bash
   docker compose restart fastapi-staging
   ```

   Y probar desde el frontend `http://localhost:${FRONTEND_STAGING_PORT}`.

   ![API UI](./dag-api-ui.jpg)

5. **Promover a Production**

   Para promover un modelo a staging disparamos el segundo DAG:

   ![Promover Prod](./dag-promote-prod.jpg)

   O via comando:

   ```bash
   docker compose exec airflow-webserver airflow dags trigger promote_model_to_production
   ```

   y reiniciar FastAPI Production para que lo cargue:

   ```bash
   docker compose restart fastapi-production
   ```

   Otra vez robar desde `http://localhost:${FRONTEND_PRODUCTION_PORT}`.

---

# Objetivos

La plataforma permite:

* almacenar datasets en un Data Lake;
* ejecutar pipelines automáticos de entrenamiento;
* registrar experimentos de Machine Learning;
* versionar modelos;
* promover modelos entre ambientes;
* exponer modelos mediante una REST API;
* validar una nueva versión del modelo antes de utilizarla en producción;
* mantener completamente aislados los consumidores de Staging y Production.

---

# Arquitectura

La solución se divide en tres dominios independientes.

```mermaid
flowchart LR

subgraph ST["Staging Environment"]
    FE_ST["Frontend<br/>Staging"]
    API_ST["FastAPI<br/>Staging"]
end

subgraph PR["Production Environment"]
    FE_PR["Frontend<br/>Production"]
    API_PR["FastAPI<br/>Production"]
end

subgraph ML["ML Platform"]

    AF["Apache Airflow"]

    MF["MLflow"]

    PG["PostgreSQL
Metadata Store"]

    subgraph DL["MinIO Data Lake"]
        RAW["Bucket
Raw Data"]
        PROC["Bucket
Processed Data"]
        ART["Bucket
Model Artifacts"]
    end

end

FE_ST --> API_ST
FE_PR --> API_PR

API_ST --> MF
API_PR --> MF

AF --> RAW
AF --> PROC
AF --> MF

MF --> PG
MF --> ART
```

## 1. Plataforma de MLOps

Es el núcleo de la solución. Es el lugar de trabajo de los ML engineers y Data Scientist.

Está formada por los siguientes servicios:

* Apache Airflow
* MLflow
* PostgreSQL
* MinIO

Todos estos componentes se encuentran conectados a una red Docker denominada:

```
ml-platform-network
```

Esta red no es accesible directamente por las aplicaciones cliente.

Su responsabilidad es:

* almacenar datos;
* entrenar modelos;
* registrar experimentos;
* administrar el ciclo de vida de los modelos.

---

## 2. Ambiente Staging

El ambiente Staging representa una aplicación que valida una nueva versión del modelo antes de publicarla en producción.

Está compuesto por:

* Frontend Staging
* FastAPI Staging

Ambos servicios pertenecen únicamente a la red:

```
staging-network
```

El servicio FastAPI Staging también posee acceso a:

```
ml-platform-network
```

para poder descargar el modelo registrado en MLflow correspondiente al ambiente Staging.

El Frontend únicamente puede comunicarse con FastAPI.

No posee acceso directo a la plataforma de ML.

---

## 3. Ambiente Production

Representa la aplicación utilizada por los usuarios finales.

Está compuesto por:

* Frontend Production
* FastAPI Production

Ambos servicios pertenecen a:

```
production-network
```

Al igual que ocurre con Staging, únicamente el servicio FastAPI tiene acceso adicional a la red de la plataforma de ML.

El Frontend nunca accede directamente a MLflow ni a MinIO.


---

# Roles y responsabilidades

La arquitectura propuesta separa claramente las responsabilidades entre el equipo encargado de desarrollar las aplicaciones consumidoras y el equipo responsable de la plataforma de Machine Learning.

Esta separación permite que la evolución de los modelos y la evolución de las aplicaciones sean procesos independientes, reduciendo el acoplamiento entre ambos equipos.

---

## Equipo de Desarrollo de Aplicaciones

Es el responsable de las aplicaciones que consumen los modelos de Machine Learning.

Administra los siguientes componentes:

* Frontend Staging
* FastAPI Staging
* Frontend Production
* FastAPI Production

Sus responsabilidades incluyen:

* desarrollar nuevas funcionalidades para las aplicaciones;
* implementar la lógica de negocio;
* definir y mantener los contratos de la API REST utilizada por los clientes;
* integrar las aplicaciones con los modelos publicados por la plataforma de ML;
* desplegar nuevas versiones de las aplicaciones en cada ambiente;
* validar la integración con nuevas versiones de modelos antes de su publicación en producción.

Este equipo **no participa del entrenamiento de modelos** ni administra el ciclo de vida de Machine Learning.

---

## Equipo de Machine Learning

Es el responsable de toda la plataforma de MLOps.

Administra los siguientes componentes:

* Apache Airflow
* MLflow
* PostgreSQL
* MinIO

Sus responsabilidades incluyen:

* construir y mantener los pipelines de DataOps y MLOps;
* desarrollar el código de entrenamiento de los modelos;
* implementar el preprocesamiento y el feature engineering;
* monitorear los experimentos de entrenamiento;
* registrar y versionar modelos mediante MLflow;
* promover modelos entre los estados Staging y Production;
* administrar los datasets y artefactos almacenados en MinIO.

Este equipo no desarrolla las aplicaciones consumidoras ni modifica su lógica de negocio.

---

## Flujo de colaboración

La interacción entre ambos equipos se produce únicamente mediante el Model Registry de MLflow.

El flujo de trabajo es el siguiente:

1. El equipo de Machine Learning entrena una nueva versión del modelo utilizando los datos almacenados en MinIO.
2. El modelo es registrado en MLflow junto con sus métricas y artefactos.
3. Una vez validado, el modelo es promovido al estado **Staging**.
4. El equipo de Desarrollo reinicia el servicio FastAPI del ambiente Staging para cargar la nueva versión del modelo y realiza las pruebas funcionales desde la aplicación cliente.
5. Si la validación es satisfactoria, el equipo de Machine Learning promueve el modelo al estado **Production**.
6. Finalmente, el servicio FastAPI de Production carga la nueva versión del modelo y queda disponible para los usuarios finales.

Esta separación permite que ambos equipos trabajen de forma independiente, compartiendo únicamente una interfaz bien definida: el modelo registrado en MLflow.

## Interacción entre equipos

```mermaid
flowchart LR

subgraph DEV["Equipo de Desarrollo"]

APP["Frontend"]

API["FastAPI"]

end

subgraph ML["Equipo de Machine Learning"]

AIR["Airflow"]

FLOW["MLflow"]

MINIO["MinIO"]

end

ML -->|"Publica modelos"| API

APP -->|"Consume API"| API

AIR --> FLOW

FLOW --> MINIO
```

# Componentes

## MinIO

MinIO simula un servicio Amazon S3.

Actualmente se usan dos buckets (`DATALAKE_BUCKET_NAME` y `MODEL_ARTIFACTS_BUCKET_NAME` en `.env`):

### Data Lake (`datalake`)

Contiene el dataset original (`raw/TMDB_movie_dataset_v11.csv`), subido por la tarea `load_data` del DAG. Estos datos nunca son modificados.

Nota: la separación en un bucket "Raw Data" y otro "Processed Data" que muestra el diagrama de arquitectura más arriba es el diseño objetivo; hoy ambos están unificados en este único bucket `datalake` (ver sección TODO).

---

### Model Artifacts (`model-artifacts`)

Es utilizado por MLflow como Artifact Store.

Aquí se almacenan:

* modelos entrenados;
* pipelines de preprocesamiento;
* matrices de confusión;
* métricas;
* cualquier artefacto generado durante el entrenamiento.

---

## PostgreSQL

PostgreSQL actúa como Backend Store de MLflow.

Aquí se almacenan:

* experimentos;
* parámetros;
* métricas;
* versiones;
* información del Model Registry.

Los archivos binarios de los modelos nunca se almacenan aquí.

---

## MLflow

MLflow administra todo el ciclo de vida del modelo.

Durante el entrenamiento registra:

* parámetros
* métricas
* artefactos

Una vez finalizado el entrenamiento, el modelo se registra dentro del Model Registry.

Ejemplo:

```
PredictionMovies

        Version 1

        Version 2

        Version 3
```

Cada versión puede tener asignado alguno de los siguientes **alias** (`MlflowClient.set_registered_model_alias`, no el mecanismo de *stages* que MLflow fue deprecando):

* `staging`
* `production`

Un alias apunta siempre a una única versión, pero puede reasignarse a otra en cualquier momento (por eso "promover" un modelo es simplemente reasignar el alias `production` a la versión que antes tenía `staging`). Los servicios FastAPI consultan siempre el modelo mediante el alias correspondiente a su ambiente (`models:/PredictionMovies@staging` / `models:/PredictionMovies@production`).

---

## Apache Airflow

Airflow es el encargado de ejecutar los pipelines de DataOps y MLOps.

El DAG principal ejecuta las siguientes tareas:

```
Carga de datos

↓

Validación

↓

Limpieza

↓

Feature Engineering

↓

Entrenamiento

↓

Evaluación

↓

Registro en MLflow
```

Cada tarea es independiente y puede ejecutarse automáticamente mediante un scheduler.

---

# Flujo de entrenamiento

El flujo completo de entrenamiento es el siguiente.

## Paso 1

Los datos son cargados al bucket Raw Data.

---

## Paso 2

Airflow inicia un DAG de entrenamiento.

---

## Paso 3

Se validan los datos.

---

## Paso 4

Se ejecuta el preprocesamiento.

---

## Paso 5

Se genera el dataset procesado.

---

## Paso 6

Se entrena el modelo.

---

## Paso 7

Se calculan las métricas.

---

## Paso 8

MLflow registra:

* parámetros
* métricas
* artefactos
* versión del modelo

---

## Paso 9

El modelo queda disponible para ser promovido al ambiente Staging o Production.

---

# Flujo de inferencia

## Ambiente Staging

El usuario accede al Frontend Staging.

```
QA Team

↓

Frontend Staging

↓

FastAPI Staging

↓

MLflow

↓

Modelo Staging

↓

Predicción
```

El endpoint FastAPI carga durante su inicialización el modelo registrado en MLflow con estado **Staging**.

Este modelo permanece cargado en memoria hasta que el servicio es reiniciado.

---

## Ambiente Production

El flujo es equivalente.

```
Usuarios

↓

Frontend Production

↓

FastAPI Production

↓

MLflow

↓

Modelo Production

↓

Predicción
```

FastAPI Production carga únicamente el modelo registrado como **Production**.

De esta forma es posible validar simultáneamente dos versiones diferentes del mismo modelo.

Por ejemplo:

```
Staging

PredictionMovies v4-RC


Production

PredictionMovies v3
```

---

# Promoción de modelos

Una vez validado un modelo en Staging, éste puede promoverse dentro de MLflow.

```
Version 4

↓

Staging

↓

Validación funcional

↓

Production
```

Posteriormente se reinicia el servicio FastAPI correspondiente para que cargue automáticamente la nueva versión del modelo.

```mermaid
stateDiagram-v2

[*] --> Registered : entrenamiento + register_model

Registered --> Staging : alias "staging" (automático, tarea promote_to_staging del DAG)

Staging --> Production : alias "production" (manual, DAG promote_model_to_production, tras validación)
```

No hay un estado "Archived" separado: un alias apunta siempre a una única versión, así que promover una nueva versión simplemente reasigna el alias correspondiente — no hace falta archivar la anterior explícitamente.

---

# Beneficios de la arquitectura

Esta arquitectura proporciona:

* separación entre DataOps y aplicaciones consumidoras;
* versionado completo de modelos;
* reproducibilidad de entrenamientos;
* almacenamiento centralizado de artefactos;
* ambientes aislados para Staging y Production;
* posibilidad de validar nuevas versiones antes de publicarlas;
* arquitectura fácilmente migrable hacia servicios administrados en la nube como Amazon S3, Amazon MWAA, SageMaker Model Registry y SageMaker Endpoints.

---

# Tecnologías utilizadas

| Componente          | Tecnología        |
| ------------------- | ----------------- |
| Orquestación        | Apache Airflow    |
| Experiment Tracking | MLflow            |
| Model Registry      | MLflow            |
| Metadata Store      | PostgreSQL        |
| Data Lake           | MinIO             |
| API REST            | FastAPI           |
| Frontend Demo       | React (o similar) |
| Contenedores        | Docker            |
| Orquestación local  | Docker Compose    |

---

# Estructura del proyecto

```
.
├── mlops-platform/
│   ├── airflow/              # Dockerfile + requirements.txt de Airflow
│   ├── dags/                 # prediction_movies_pipeline.py, promote_model.py
│   ├── ml/                   # training/, preprocessing/, inference/ (importable como `from ml import ...`)
│   ├── mlflow/                # Dockerfile + requirements.txt del server de MLflow
│   ├── postgresql/           # Dockerfile + init/ (crea la DB de Airflow además de la de MLflow)
│   └── dataset/               # TMDB_movie_dataset_v11.csv, versionado con Git LFS
├── client-app/
│   ├── fastapi/               # una sola app, reusada para Staging/Production vía env vars
│   └── react/                 # ídem, vía build args
├── docker-compose.yml
├── .env.template
└── README.md
```

`fastapi-staging`/`fastapi-production` y `frontend-staging`/`frontend-production` **no** son directorios separados: son la misma imagen de `client-app/fastapi`/`client-app/react`, instanciada dos veces en `docker-compose.yml` y diferenciada por variables de entorno (`MODEL_ALIAS`) y build args (`VITE_API_URL`).

---

# Configuración

## Variables de entorno

La plataforma se configura mediante un archivo `.env` en la raíz del repositorio, leído por Docker Compose. Para crear el propio:

```bash
cp .env.template .env
```

`.env.template` (versionado en git) documenta todas las variables necesarias con valores de ejemplo. Son todas credenciales demo para desarrollo local (MinIO, Postgres, Airflow) y pueden dejarse como están.

## Dataset

`TMDB_movie_dataset_v11.csv` está versionado directamente en el repositorio, en `mlops-platform/dataset/`, usando [Git LFS](https://git-lfs.com/) (pesa ~600MB, demasiado para versionarlo como blob normal de git). Por eso hace falta tener `git-lfs` instalado (`git lfs install`, una única vez por máquina) **antes** de clonar o hacer `git pull` — si el archivo aparece como un texto corto tipo `version https://git-lfs.github.com/spec/v1 ...` en vez del CSV real, es que faltó ese paso: correr `git lfs pull` para traer el contenido real.

La tarea `load_data` del DAG (`mlops-platform/dags/prediction_movies_pipeline.py`) sube ese archivo al bucket Raw Data de MinIO la primera vez que se corre el DAG (si el archivo ya está en MinIO, no lo vuelve a subir). Esto no pasa solo: hay que disparar el DAG manualmente (UI via http://localhost:8080 o `airflow dags trigger prediction_movies_pipeline`) y, si es la primera vez que corre, despausarlo antes (manualmente en la UI o `airflow dags unpause prediction_movies_pipeline`), porque todo DAG nuevo arranca pausado en Airflow.

---

# Future Work

• Model Monitoring

• Data Drift Detection

• Concept Drift

• Automated Retraining

• Canary Deployments

---

# TODO

El esqueleto de contenedores (Docker Compose, redes, Dockerfiles) ya está armado. El código de todo el pipeline (`ml/preprocessing`, `ml/training`, `ml/inference`, el DAG, `/predict`) está implementado y conectado de punta a punta, pero **no hay todavía una corrida end-to-end confirmada** en un entorno con recursos suficientes (en un Mac de 8GB el DAG llegó a matar por OOM la tarea `validate_data`, que sólo lee el CSV completo; no se llegó a correr `feature_engineering`/`train_model`/`register_model`). Este es el estado real:

- [x] Portar la limpieza y validación de datos del notebook (`prediction_movies_imdb.ipynb`) a `mlops-platform/ml/preprocessing`.
- [x] Portar el feature engineering (encoding de `original_language`, `genres`, `production_countries`, `production_companies`) a `mlops-platform/ml/preprocessing`, con el top-N y los `MultiLabelBinarizer` fiteados únicamente sobre el split de entrenamiento y persistidos como artefacto (`FeatureEncoder`) para que `ml/inference` reproduzca el mismo encoding.
- [x] Portar el entrenamiento y la evaluación del modelo a `mlops-platform/ml/training`.
- [x] Implementar el registro del modelo entrenado (`register_model`) y su promoción a alias (`promote_model`, `get_model_version_by_alias`) en el Model Registry de MLflow, bajo el nombre `PredictionMovies`.
- [x] Versionar `TMDB_movie_dataset_v11.csv` en el repo (`mlops-platform/dataset/`, vía Git LFS) y completar la tarea `load_data` del DAG (confirmado funcionando: sube el archivo a MinIO y lo descarga).
- [x] Conectar las tareas del DAG (`mlops-platform/dags/prediction_movies_pipeline.py`, `promote_model.py`) con las funciones reales de `ml/`.
- [x] Correr el DAG de punta a punta en un entorno con recursos suficientes y confirmar que el modelo `PredictionMovies` queda registrado en MLflow con el alias `staging` asignado. `load_data` y `validate_data` corrieron OK; el resto del pipeline no se validó todavía.
- [x] Validar `fastapi-staging` con un modelo real en Staging (depende del punto anterior).
- [x] Promover el modelo a Production una vez validado y confirmar que `fastapi-production` lo carga correctamente.
- [x] Implementar `ml/inference` (carga del modelo + del `FeatureEncoder` persistido + predicción) y conectarlo al endpoint `/predict` de FastAPI.
- [x] Reemplazar el frontend placeholder por un formulario real que envíe `budget`, `runtime`, `original_language`, `genres`, `production_countries` y `production_companies` a `/predict`.
- [ ] (nice-to-have) Separar el bucket "Processed Data" del bucket "Raw Data" en MinIO, tal como describe la arquitectura de este documento (hoy están unificados en un solo bucket `datalake`).
- [ ] Rotar el `KAGGLE_API_TOKEN` que quedó en `.env` de cuando el DAG todavía descargaba el dataset de Kaggle: ya no se usa (el dataset se versiona en el repo, ver "Dataset"), pero sigue siendo una credencial real expuesta en el historial de git.
