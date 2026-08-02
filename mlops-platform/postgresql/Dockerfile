FROM postgres:16-alpine

# Scripts de inicialización: se ejecutan una única vez, la primera vez
# que el volumen de datos de Postgres se crea vacío.
COPY init/ /docker-entrypoint-initdb.d/
