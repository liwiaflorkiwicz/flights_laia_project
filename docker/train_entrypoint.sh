#!/bin/bash
set -e

echo "1. Pulling dataset via DVC from VM1 ==="
dvc pull data/processed/flights_2022_2023.parquet --force

echo "2. Preparing temporal data splits ==="
python -m src.data.prepare_splits --config configs/train.yaml

echo "3. Training model and registering in MLflow ==="
python -m src.modeling.train "$@"