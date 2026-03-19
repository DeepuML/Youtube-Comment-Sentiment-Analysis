# Youtube-Comment-Sentiment-Analysis

A Chrome extension that detects the sentiment of YouTube comments in real-time using a trained machine learning model.  
The project is fully automated with **MLflow** for experiment tracking & model registry, **DVC** for data and model versioning, and a **CI/CD pipeline** for continuous training and deployment.

---

## 📌 Features

- **Data Pipeline (DVC)** – Ingests real YouTube comment data, preprocesses it, and trains a sentiment model end-to-end.
- **Sentiment Model** – Trained LightGBM model for positive/negative/neutral comment classification.
- **MLflow Integration** – Experiment tracking, model versioning, and registry.
- **Automated Model Registration** – Models are automatically pushed to MLflow Registry with stage/alias tagging.
- **Chrome Extension** – Displays comment sentiment directly on YouTube pages.
- **CI/CD Pipeline** – GitHub Actions workflow runs the full pipeline (`dvc repro`) and updates the deployed model.

---

## 📂 Project Organization

```
├── LICENSE
├── Makefile                    <- Makefile with commands like `make data`
├── README.md                   <- Top-level README for developers
├── appsec.yml                  <- AWS CodeDeploy AppSpec file
├── Dockerfile                  <- Docker image for the Flask API
├── dvc.yaml                    <- DVC pipeline definition
├── params.yaml                 <- Hyperparameters and pipeline configuration
├── requirements.txt            <- Python dependencies for the ML pipeline
├── setup.py                    <- Makes project pip-installable (pip install -e .)
├── tox.ini                     <- Tox configuration
│
├── data/                       <- Data directory (managed by DVC, excluded from git)
│   ├── raw/                    <- Raw comment data downloaded from the configured URL
│   └── interim/                <- Preprocessed data ready for model training
│
├── deploy/
│   └── scripts/
│       ├── install_dependencies.sh   <- EC2 setup script (Docker + AWS CLI)
│       └── start_docker.sh           <- Pulls and starts the Docker container on EC2
│
├── flask_app/
│   ├── app.py                  <- Flask REST API (sentiment prediction endpoints)
│   └── requirements.txt        <- Flask app Python dependencies
│
├── models/                     <- Placeholder for locally stored models
│
├── notebooks/                  <- Jupyter notebooks for exploration and experiments
│
├── references/                 <- Data dictionaries and explanatory materials
│
├── reports/
│   └── figures/                <- Generated plots and confusion matrices
│
├── scripts/
│   ├── mlflow_test.py          <- Smoke-test for MLflow connectivity
│   ├── promote_model.py        <- Promotes the staging model to production
│   ├── test_flask_api.py       <- pytest tests for Flask API endpoints
│   ├── test_load_model.py      <- pytest test: model loads from MLflow registry
│   ├── test_model_performance.py <- pytest test: model meets accuracy thresholds
│   └── test_model_signature.py <- pytest test: model signature is correct
│
└── src/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── data_ingestion.py       <- Downloads & splits YouTube comment data
    │   └── data_preprocessing.py  <- Cleans and normalises comment text
    ├── features/
    │   └── __init__.py
    ├── model/
    │   ├── __init__.py
    │   ├── model_building.py       <- TF-IDF vectorisation + LightGBM training
    │   ├── model_evaluation.py     <- Evaluation metrics, confusion matrix, MLflow logging
    │   └── register_model.py      <- Registers model in MLflow Model Registry
    └── visualization/
        └── __init__.py
```

---

## 🚀 How It Works

1. **Data Ingestion** (`src/data/data_ingestion.py`)
   - Downloads YouTube comment data from the URL configured in `params.yaml`
   - Normalizes column names, removes duplicates and missing values
   - Splits into train/test sets and saves to `data/raw/`

2. **Preprocessing** (`src/data/data_preprocessing.py`)
   - Converts text to lowercase, removes noise, strips stopwords (preserving negation words)
   - Lemmatises tokens
   - Saves processed data to `data/interim/`

3. **Model Training** (`src/model/model_building.py`)
   - Applies TF-IDF vectorisation (configurable n-grams and feature count)
   - Trains a LightGBM multiclass classifier
   - Saves model and vectoriser to the project root

4. **Model Evaluation** (`src/model/model_evaluation.py`)
   - Generates classification report and confusion matrix
   - Logs all metrics, parameters, and artifacts to MLflow

5. **Model Registration** (`src/model/register_model.py`)
   - Registers the trained model in the MLflow Model Registry
   - Transitions it to the `Staging` stage

6. **CI/CD Pipeline** (`.github/workflows/cicd.yaml`)
   - Triggered on every push to `master`
   - Runs `dvc repro`, runs model validation tests, promotes to Production, then builds and pushes a Docker image to ECR and deploys via AWS CodeDeploy

7. **Chrome Extension Integration**
   - Calls the Flask API (`/predict`) with YouTube comment text
   - Highlights comments with sentiment color codes (green/gray/red)

---

## ⚙️ Running the Project Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/Youtube-Comment-Sentiment-Analysis.git
cd Youtube-Comment-Sentiment-Analysis

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set required environment variables
export MLFLOW_TRACKING_URI="http://<your-mlflow-host>:5000"

# 5. Reproduce the DVC pipeline
dvc repro

# 6. Push tracked artifacts to remote storage
dvc push
```

### Running the Flask API locally

```bash
cd flask_app
export MLFLOW_TRACKING_URI="http://<your-mlflow-host>:5000"
export MODEL_VERSION="Production"          # or a specific version number
python app.py
# API available at http://localhost:5000
```

### Running with Docker

```bash
docker build -t youtube-comment-analysis .
docker run -p 5000:5000 \
  -e MLFLOW_TRACKING_URI="http://<your-mlflow-host>:5000" \
  -e MODEL_VERSION="Production" \
  youtube-comment-analysis
```

---

## 🔧 Configuration (`params.yaml`)

| Parameter | Description | Default |
|---|---|---|
| `data_ingestion.test_size` | Fraction of data reserved for testing | `0.2` |
| `data_ingestion.data_url` | URL of the YouTube comments CSV dataset | see params.yaml |
| `model_building.ngram_range` | TF-IDF n-gram range | `[1, 3]` |
| `model_building.max_features` | Maximum TF-IDF vocabulary size | `10000` |
| `model_building.learning_rate` | LightGBM learning rate | `0.08` |
| `model_building.max_depth` | LightGBM max tree depth | `20` |
| `model_building.n_estimators` | Number of boosting rounds | `367` |

---

## 🔐 Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3, ECR, and CodeDeploy |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key |
| `AWS_ACCOUNT_ID` | AWS account ID (used for ECR URI construction) |
| `MLFLOW_TRACKING_URI` | Full URI of the MLflow tracking server |

---

## 📊 Sentiment Classes

| Label | Meaning |
|---|---|
| `1` | Positive |
| `0` | Neutral |
| `-1` | Negative |
