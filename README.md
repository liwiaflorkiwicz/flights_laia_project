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

---

## System Architecture

The project is structured across two Virtual Machines and GitHub Actions:

- **VM1 (`10.17.0.190`)**: Hosts the central **MLflow Tracking Server** (port 5050) 
and **DVC SSH Remote Storage** (`/home/admin/dvc_storage`).
- **VM2 (`10.17.0.216`)**: Hosts the **GitHub Actions Self-Hosted Runner**
and the production **FastAPI Service** inside a Docker container (port 8000).

## Repository layout

```
.github/workflows/        # CI/CD GitHub Actions pipelines
    continuous_integration.yml
    continuous_delivery.yml
configs/
  train.yaml              # data, split, features, model and MLflow settings
docker/
    Dockerfile              # Python 3.13-slim image for production API serving
    docker-compose.yml        # Docker service orchestration & environment setup
src/
  api/
    main.py               # FastAPI application lifecycle and endpoints
    schemas.py            # Pydantic input/output payload validation
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

Numeric features: day-of-month and day-of-week (with a weekend flag and
cyclical encodings), scheduled departure/arrival time (hour, minute, minutes
since midnight, red-eye, overnight), scheduled duration, distance, implied
speed, schedule buffer and same-state route.

**Year-specific seasonality is deliberately excluded.** `MONTH`, `DAY_OF_YEAR`,
`WEEK_OF_YEAR`, `QUARTER`, `month_sin` and `month_cos` were tested and then
dropped: with only 2022–2023 data they effectively memorise the weather of
those specific months. On a temporally held-out December test set they caused
severe delay over-prediction. Removing them raised test macro-F1 from 0.351 to
0.389 and on-time recall from 0.63 to 0.86 (see
`reports/feature_ablation.json`). Day-of-month and day-of-week are kept because
they capture holidays and weekly patterns without tying the model to a
particular year.

Categorical features: operating carrier, origin/destination airport and
state, departure time block, route. They are ordinal-encoded with
`handle_unknown="use_encoded_value"` (`-1` for unseen categories) and
augmented with training-frequency encodings (unseen -> `0.0`). This means an
airline or airport never seen during training cannot crash inference.

## Model

`LightGBM` multiclass classifier with **mild inverse-frequency class
weighting**, early stopping on a custom **macro-F1** metric, and a single
scikit-learn `Pipeline` that contains both feature engineering and the
classifier. The whole pipeline is saved as one artifact, so serving receives
raw BTS records.

Class weights are `(N / (K * n_c)) ** power`, rescaled to a prevalence-weighted
mean of 1, with `power` configurable (`model.class_weight_power`). `power=0`
means no weighting and `power=1` reproduces sklearn's `class_weight="balanced"`.
Full balancing over-predicts delays and collapsed on-time recall to 0.41 on the
December test set. `power=0.75` is the chosen compromise: it keeps minority
class recall while avoiding the over-prediction (see
`reports/class_weight_experiments_reduced.json`).

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

## Results

Final temporal model (11.55M training flights, 35 features, weighting power
0.75, early stopping at iteration 21):

| Split | Macro-F1 | Accuracy | On-time recall | Moderate recall | Severe recall |
| ----- | -------- | -------- | -------------- | --------------- | ------------- |
| val (Oct–Nov 2023) | 0.394 | 0.766 | 0.868 | 0.174 | 0.158 |
| test (Dec 2023)     | 0.389 | 0.749 | 0.858 | 0.159 | 0.169 |

For comparison, the initial model with full `class_weight="balanced"` and
year-specific seasonality scored val macro-F1 0.383 / test 0.295, with a test
on-time recall of only 0.415 because it over-predicted delays. The two changes
(mild inverse-frequency weighting and dropping year-specific seasonality)
raised test macro-F1 by ~0.09 and restored on-time recall.

The remaining error is almost entirely confusion between `moderate_delay` and
`severe_delay`; with only pre-departure information the exact arrival delay is
highly uncertain. Aggregate predicted delay share now tracks the true delay
share closely, so the model no longer systematically over-predicts delays.

## Outputs

- `models/model.joblib` — complete preprocessing + classifier pipeline
- `models/metadata.json` — versions, features, split summary, metrics, config
- `reports/metrics.json` — validation/test overall, per-class and subgroup metrics
- `reports/classification_report.txt` — sklearn reports
- `reports/figures/confusion_matrix_{val,test}.png`
- `reports/feature_importance.csv`

## Deployment / monitoring

The pipeline artifact (`models/model.joblib`) is fully integrated into the containerized 
FastAPI `/classify` endpoint, receiving raw BTS field names in JSON format and returning 
ordered `classifications` predictions. 

With containerized deployment and automated CI/CD established, the primary operational focus 
moves to post-deployment monitoring:
- **Subgroup Monitoring:** Per-airline and per-origin-airport evaluation metrics generated 
during training (`reports/metrics.json`) serve as the established baseline for detecting operational skew.
- **Drift & Degradation Checks:** Tracking production inference request payloads against 
baseline distributions to detect data drift and performance degradation on VM2 before initiating 
automated retrains via Continuous Delivery (`continuous_delivery.yml`).

## CI/CD Automation Pipeline

Automated workflows are managed via GitHub Actions:

1. **Continuous Integration (`1_continuous_integration.yml`)**:
   - Runs on `ubuntu-latest` upon `push` and `pull_request`.
   - Executes `flake8` syntax checks, generates a lightweight dummy model for quick smoke testing, runs `pytest`, and builds the Docker image.
2. **Continuous Delivery (`2_continuous_delivery.yml`)**:
   - Runs on `self-hosted` runner (VM2) upon `push` to `main` or manual trigger.
   - Securely pulls remote data via DVC SSH (`dvc pull`), runs temporal data splitting, trains the LightGBM model, logs runs to MLflow (`CD-${SHA}`),
   builds the Docker image.

## Deployment & Verification Status

The project checkpoint requirements are fully implemented and verified across the dual-VM architecture:

| Requirement | Implementation Details                                                                                      | Status |
| :--- |:------------------------------------------------------------------------------------------------------------| :--- |
| **Baseline Model** | LightGBM multiclass model with inverse-frequency weighting (`power=0.75`)                                   | Verified |
| **Code & Data Versioning** | Git repository paired with DVC tracking over SSH (`vm1_storage`)                                            | Verified |
| **Experiment Tracking** | MLflow tracking server running on VM1 (`10.17.0.190:5050`)                                                  | Verified |
| **CI/CD Automation** | GitHub Actions workflows (`continuous_integration.yml` on push/PR, `continuous_delivery.yml` on VM2 runner) | Verified |
| **Containerized Deployment** | FastAPI application served inside Docker container (`flight_api_prod`)                                      | Verified |
| **Endpoint Registration** | `POST /classify` and interactive OpenAPI documentation at `/docs` (VM2: `10.17.0.216:8000`)                 | Verified |
