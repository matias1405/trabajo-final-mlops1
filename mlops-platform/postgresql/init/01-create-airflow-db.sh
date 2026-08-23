#!/bin/sh
# POSTGRES_DB (variable de entorno del contenedor) ya crea la base de datos
# usada como Backend Store de MLflow. Este script agrega la base de datos
# separada que utiliza Airflow como Metadata Store, reutilizando el mismo
# usuario (POSTGRES_USER) para simplificar el esquema de credenciales.
#
# Va como .sh (no .sql) porque docker-entrypoint-initdb.d no interpola
# variables de entorno dentro de archivos .sql; los .sh sí las heredan.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE "$AIRFLOW_DB_NAME";
EOSQL
