FROM postgres:16-alpine@sha256:cf78e76683b9ca8c5733cbbdce6c9262b45b6767934dd0a95e671f9a0fc20685 

# Scripts de inicialización: se ejecutan una única vez, la primera vez
# que el volumen de datos de Postgres se crea vacío.
COPY init/ /docker-entrypoint-initdb.d/
