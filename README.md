# flights_laia_project

Flight Delay Severity Prediction — end-to-end MLOps pipeline for the LAIA
project (2026/27).

The task is to classify each US domestic flight into one of three arrival
delay severity classes using **only information available before scheduled
departure**:

| Category         | Definition                          |
| ---------------- | ----------------------------------- |
| `on_time`        | arrival delay < 15 min              |
| `moderate_delay` | 15 min <= arrival delay < 60 min    |
| `severe_delay`   | arrival delay >= 60 min             |

## Repository layout

```
configs/
  train.yaml              # data, split, features, model and MLflow settings
src/
  data/
    make_dataset.py       # raw BTS CSV -> processed Parquet (DVC stage)
    prepare_splits.py     # memory-safe temporal/stratified splits
  features/
    build_features.py     # leakage-free feature engineering + encoders
  modeling/
    train.py              # training, MLflow logging, artifact generation
    evaluate.py           # overall / per-class / subgroup metrics
  utils/
    config.py, dates.py
tests/                    # pytest unit and end-to-end smoke tests
data/processed/           # DVC-tracked processed data and splits
models/                   # joblib pipeline + metadata (generated)
reports/                  # metrics, classification report, figures
```

## Data and leakage rules

Input: `data/processed/flights_2022_2023.parquet` (13,275,415 flights).

The preprocessing stage keeps only flights that were neither cancelled nor
diverted and that have a valid arrival delay. The target is built from
`ARR_DELAY`, then the target/filter columns are dropped.

**Only pre-departure fields are used.** Post-departure fields such as
`DepDelay`, `DepTime`, `TaxiOut`, `AirTime`, `ActualElapsedTime` and the delay
cause columns are never read by the feature pipeline.

## Split strategy

The default split is **temporal**, because the centralised evaluation always
scores the model on future flights:

| Split | Period                | Rows       |
| ----- | --------------------- | ---------- |
| train | 2022-01-01 – 2023-09-30 | 11,550,058 |
| val   | 2023-10-01 – 2023-11-30 | 1,158,416  |
| test  | 2023-12-01 – 2023-12-31 | 566,941    |

The test set is touched exactly once, after model selection. A random
`stratified` split is implemented for ablation/diagnostics only: shuffling
flights leaks future periods into training and is not representative of the
deployment setting.

Class shares change across periods (e.g. `severe_delay` 7.3% in train vs
4.5% in test), which is precisely why macro-F1 and subgroup metrics are
reported instead of accuracy alone.

## Feature engineering

Numeric features: calendar parts (month, day, weekday, day-of-year, week,
quarter, weekend), cyclical encodings, scheduled departure/arrival time
(hour, minute, minutes since midnight, red-eye, overnight), scheduled
duration, distance, implied speed, schedule buffer and same-state route.

Categorical features: operating carrier, origin/destination airport and
state, departure time block, route. They are ordinal-encoded with
`handle_unknown="use_encoded_value"` (`-1` for unseen categories) and
augmented with training-frequency encodings (unseen -> `0.0`). This means an
airline or airport never seen during training cannot crash inference.

## Model

`LightGBM` multiclass classifier with balanced class weights, early stopping
on a custom **macro-F1** metric, and a single scikit-learn `Pipeline` that
contains both feature engineering and the classifier. The whole pipeline is
saved as one artifact, so serving receives raw BTS records.

## Reproduce

```bash
pip install -r requirements.txt

# 1) Optional: rebuild processed Parquet with DVC
dvc repro preprocess

# 2) Build train/val/test splits (streams the 13M-row Parquet)
python -m src.data.prepare_splits --config configs/train.yaml

# 3) Train, evaluate and log to MLflow
python -m src.modeling.train --config configs/train.yaml

# Quick local smoke run on a stratified subsample
python -m src.modeling.train --config configs/train.yaml \
  --max-train-rows 200000 --no-mlflow

# Ablation: random stratified split (diagnostics only)
python -m src.data.prepare_splits --config configs/train.yaml --strategy stratified
```

MLflow uses the local `mlruns/` directory by default. Set
`MLFLOW_TRACKING_URI` (for example to the VM1 tracking server) to override it.
Enable registry registration with `--register` after the tracking server uses
a database backend.

## Tests

```bash
python -m pytest tests -q
```

Tests cover date parsing, temporal boundaries, stratified proportions,
leakage-column exclusion, unseen categories, minimal/malformed API payloads
and an end-to-end fit/predict smoke test.

## Outputs

- `models/model.joblib` — complete preprocessing + classifier pipeline
- `models/metadata.json` — versions, features, split summary, metrics, config
- `reports/metrics.json` — validation/test overall, per-class and subgroup metrics
- `reports/classification_report.txt` — sklearn reports
- `reports/figures/confusion_matrix_{val,test}.png`
- `reports/feature_importance.csv`

## Next steps (deployment / monitoring)

The same pipeline artifact is intended for the FastAPI `/classify` endpoint
(raw BTS field names in, ordered `classifications` out), and for subgroup
monitoring by operating airline and origin airport. The per-airline/per-airport
metrics produced here form the reference for future drift and degradation
checks.
