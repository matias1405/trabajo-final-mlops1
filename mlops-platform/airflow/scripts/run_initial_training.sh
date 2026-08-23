#!/bin/bash

set -e

DAG_V1="train-model-prediction-movies-v1"
DAG_V2="train-model-prediction-movies-v2"

echo "========================================"
echo "Waiting for DAGs to be available..."
echo "========================================"

for DAG_ID in "$DAG_V1" "$DAG_V2"; do
    until airflow dags list | grep -q "$DAG_ID"; do
        echo "Waiting for $DAG_ID..."
        sleep 5
    done

    echo "$DAG_ID is available."
done

echo ""
echo "========================================"
echo "Triggering V1"
echo "========================================"

airflow dags trigger "$DAG_V1"

echo ""
echo "========================================"
echo "Triggering V2"
echo "========================================"

airflow dags trigger "$DAG_V2"

echo ""
echo "========================================"
echo "Waiting for both DAGs to finish..."
echo "========================================"

while true; do

    V1_STATUS=$(airflow dags list-runs \
        --dag-id "$DAG_V1" \
        --output json |
        python -c "
import sys
import json

runs = json.load(sys.stdin)

if runs:
    print(runs[0]['state'])
")

    V2_STATUS=$(airflow dags list-runs \
        --dag-id "$DAG_V2" \
        --output json |
        python -c "
import sys
import json

runs = json.load(sys.stdin)

if runs:
    print(runs[0]['state'])
")

    echo "V1 status: $V1_STATUS"
    echo "V2 status: $V2_STATUS"

    # Si cualquiera falla, todo el bootstrap falla.
    if [ "$V1_STATUS" = "failed" ]; then
        echo "ERROR: $DAG_V1 failed."
        exit 1
    fi

    if [ "$V2_STATUS" = "failed" ]; then
        echo "ERROR: $DAG_V2 failed."
        exit 1
    fi

    # Ambos terminaron correctamente.
    if [ "$V1_STATUS" = "success" ] && [ "$V2_STATUS" = "success" ]; then
        echo ""
        echo "========================================"
        echo "Both training DAGs completed successfully."
        echo "========================================"

        exit 0
    fi

    sleep 5
done