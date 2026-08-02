-- POSTGRES_DB (variable de entorno del contenedor) ya crea la base de datos
-- usada como Backend Store de MLflow. Este script agrega la base de datos
-- separada que utiliza Airflow como Metadata Store, reutilizando el mismo
-- usuario (POSTGRES_USER) para simplificar el esquema de credenciales.
CREATE DATABASE airflow;
