# 🎬 YouTube Comment Sentiment Analysis

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0.3-black?logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/MLflow-2.15.0-blue?logo=mlflow&logoColor=white" alt="MLflow"/>
  <img src="https://img.shields.io/badge/DVC-3.53.0-945DD6?logo=dvc&logoColor=white" alt="DVC"/>
  <img src="https://img.shields.io/badge/LightGBM-4.5.0-green" alt="LightGBM"/>
  <img src="https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white" alt="Docker"/>
  <img src="https://img.shields.io/badge/AWS-ECR%20%7C%20CodeDeploy-FF9900?logo=amazonaws&logoColor=white" alt="AWS"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"/>
</p>

<p align="center">
  A <strong>Chrome extension</strong> powered by a machine learning backend that detects the sentiment of YouTube comments in real-time.<br/>
  Fully automated with <strong>MLflow</strong> for experiment tracking, <strong>DVC</strong> for data versioning, and a <strong>CI/CD pipeline</strong> for continuous training and deployment to AWS.
</p>

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🛠️ Tech Stack](#️-tech-stack)
- [📂 Project Structure](#-project-structure)
- [🔄 ML Pipeline](#-ml-pipeline)
- [🌐 Flask API Endpoints](#-flask-api-endpoints)
- [🧩 Chrome Extension](#-chrome-extension)
- [⚙️ Local Setup & Installation](#️-local-setup--installation)
- [🐳 Docker Deployment](#-docker-deployment)
- [🚀 CI/CD Pipeline](#-cicd-pipeline)
- [📊 Model Performance](#-model-performance)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔁 **Data Pipeline (DVC)** | Reproducible pipeline for data ingestion, preprocessing, and model training |
| 🤖 **Sentiment Model** | LightGBM classifier for **Positive / Neutral / Negative** comment classification |
| 📈 **MLflow Integration** | Full experiment tracking, model versioning, and centralized model registry |
| 📦 **Automated Model Registration** | Models auto-pushed to MLflow Registry with environment tagging |
| 🧩 **Chrome Extension** | Real-time sentiment overlay on YouTube comment sections |
| 🚀 **CI/CD Pipeline** | GitHub Actions → DVC repro → Tests → Docker → AWS ECR + CodeDeploy |
| ☁️ **Cloud Deployment** | Containerized Flask API deployed to AWS EC2 via ECR & CodeDeploy |
| 📊 **Visual Analytics** | Sentiment pie charts, word clouds, and monthly trend graphs |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **ML Framework** | LightGBM, scikit-learn |
| **NLP** | NLTK (tokenization, stopwords, lemmatization), TF-IDF |
| **Experiment Tracking** | MLflow |
| **Data Versioning** | DVC + AWS S3 |
| **API Server** | Flask + Flask-CORS |
| **Visualization** | Matplotlib, WordCloud |
| **Containerization** | Docker |
| **CI/CD** | GitHub Actions |
| **Cloud** | AWS ECR, AWS CodeDeploy, AWS S3, AWS EC2 |
| **Testing** | pytest |

---

## 📂 Project Structure

```
Youtube-Comment-Sentiment-Analysis/
├── .github/
│   └── workflows/
│       └── cicd.yaml              ← GitHub Actions CI/CD pipeline
├── deploy/
│   └── scripts/                   ← Deployment scripts (install, start docker)
├── flask_app/
│   ├── app.py                     ← Flask API server
│   └── requirements.txt
├── notebooks/                     ← Exploratory Jupyter notebooks
├── references/                    ← Data dictionaries and manuals
├── reports/
│   └── figures/                   ← Generated charts and confusion matrices
├── scripts/
│   ├── test_load_model.py         ← Model loading test
│   ├── test_model_signature.py    ← Model signature test
│   ├── test_model_performance.py  ← Model performance test
│   └── promote_model.py           ← Promote model to production
├── src/
│   ├── data/
│   │   ├── data_ingestion.py      ← Fetch & split raw YouTube comment data
│   │   └── data_preprocessing.py  ← Clean & normalize text
│   └── model/
│       ├── model_building.py      ← Train LightGBM with TF-IDF features
│       ├── model_evaluation.py    ← Evaluate & log metrics to MLflow
│       └── register_model.py      ← Register model in MLflow Registry
├── dvc.yaml                       ← DVC pipeline definition
├── params.yaml                    ← Hyperparameters and pipeline config
├── Dockerfile                     ← Docker image for Flask API
├── requirements.txt               ← Python dependencies
└── README.md
```

---

## 🔄 ML Pipeline

The pipeline is defined in `dvc.yaml` and can be fully reproduced with a single command: `dvc repro`.

```
Data Ingestion ──► Preprocessing ──► Model Building ──► Model Evaluation ──► Model Registration
```

### Stage Details

| # | Stage | Script | Output |
|---|---|---|---|
| 1 | **Data Ingestion** | `src/data/data_ingestion.py` | `data/raw/train.csv`, `data/raw/test.csv` |
| 2 | **Preprocessing** | `src/data/data_preprocessing.py` | `data/interim/train_processed.csv`, `data/interim/test_processed.csv` |
| 3 | **Model Building** | `src/model/model_building.py` | `lgbm_model.pkl`, `tfidf_vectorizer.pkl` |
| 4 | **Model Evaluation** | `src/model/model_evaluation.py` | `experiment_info.json`, MLflow run logged |
| 5 | **Model Registration** | `src/model/register_model.py` | Model registered in MLflow Registry |

### Hyperparameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.2

model_building:
  ngram_range: [1, 3]
  max_features: 10000
  learning_rate: 0.08
  max_depth: 20
  n_estimators: 367
```

---

## 🌐 Flask API Endpoints

The Flask backend (`flask_app/app.py`) serves the model and provides visualization endpoints. The server runs on port **3000**.

| Method | Endpoint | Description | Request Body |
|---|---|---|---|
| `GET` | `/` | Health check | — |
| `POST` | `/predict` | Predict sentiment for a list of comments | `{ "comments": ["great video!", ...] }` |
| `POST` | `/predict_with_timestamps` | Predict sentiment with timestamps | `{ "comments": [{ "text": "...", "timestamp": "..." }] }` |
| `POST` | `/generate_chart` | Generate a sentiment pie chart (PNG) | `{ "sentiment_counts": { "1": 50, "0": 30, "-1": 20 } }` |
| `POST` | `/generate_wordcloud` | Generate a word cloud image (PNG) | `{ "comments": ["great video!", ...] }` |
| `POST` | `/generate_trend_graph` | Generate a monthly sentiment trend graph (PNG) | `{ "sentiment_data": [{ "sentiment": "1", "timestamp": "..." }] }` |

### Sentiment Labels

| Label | Meaning |
|---|---|
| `1` | 😊 Positive |
| `0` | 😐 Neutral |
| `-1` | 😠 Negative |

#### Example: `/predict` Request

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"comments": ["This video is amazing!", "Worst content ever", "It was okay"]}'
```

**Response:**

```json
[
  { "comment": "This video is amazing!", "sentiment": "1"  },
  { "comment": "Worst content ever",     "sentiment": "-1" },
  { "comment": "It was okay",            "sentiment": "0"  }
]
```

---

## 🧩 Chrome Extension

The Chrome extension connects to the deployed Flask API and provides real-time sentiment analysis directly on YouTube pages:

- 🟢 **Green** highlight — Positive comments
- ⚪ **Grey** highlight — Neutral comments
- 🔴 **Red** highlight — Negative comments
- 📊 Sentiment distribution **pie chart** overlay
- ☁️ **Word cloud** of frequent terms in comments
- 📈 **Monthly trend graph** showing sentiment over time

> The extension reads visible comments from the active YouTube page and sends them to the `/predict` endpoint in real-time, displaying color-coded sentiment badges next to each comment.

---

## ⚙️ Local Setup & Installation

### Prerequisites

- Python 3.10
- [Conda](https://docs.conda.io/en/latest/) or `venv`
- AWS credentials (for DVC remote storage)
- MLflow tracking server (local or remote)

### 1. Clone the Repository

```bash
git clone https://github.com/DeepuML/Youtube-Comment-Sentiment-Analysis.git
cd Youtube-Comment-Sentiment-Analysis
```

### 2. Create & Activate Virtual Environment

```bash
conda create -n yt_project python=3.10
conda activate yt_project
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure AWS Credentials (for DVC remote)

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 5. Reproduce the Full ML Pipeline

```bash
dvc repro
```

### 6. Push Artifacts to Remote Storage

```bash
dvc push
```

### 7. Run the Flask API Locally

```bash
cd flask_app
python app.py
# API will be available at http://localhost:3000
```

---

## 🐳 Docker Deployment

Build and run the Flask API as a self-contained Docker container:

```bash
# Build the image
docker build -t youtube_comment_analysis .

# Run the container
docker run -p 3000:3000 youtube_comment_analysis
```

The API will be available at `http://localhost:3000`.

---

## 🚀 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/cicd.yaml`) triggers automatically on every push to the `master` branch and executes the full automated pipeline:

```
Push to master
      │
      ▼
┌──────────────────────┐
│  Install Dependencies │
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│     dvc repro         │  ← Run full ML pipeline
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│     dvc push          │  ← Push artifacts to S3
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│   Run pytest Tests    │  ← Load, signature & performance tests
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│  Promote to Production│  ← Tag model in MLflow Registry
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│  Build Docker Image   │
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│   Push to AWS ECR     │
└───────────┬──────────┘
            │
            ▼
┌──────────────────────┐
│ Deploy via CodeDeploy │  ← Deploy to EC2
└──────────────────────┘
```

### Required GitHub Secrets

| Secret | Description |
|---|---|
| `AWS_ACCESS_KEY_ID` | AWS access key for S3, ECR, and CodeDeploy |
| `AWS_SECRET_ACCESS_KEY` | AWS secret access key |
| `GITHUB_TOKEN` | Auto-provided by GitHub Actions |

---

## 📊 Model Performance

The LightGBM model is evaluated on a held-out test set. Metrics and confusion matrices are logged to MLflow and saved in the `reports/` directory.

![Confusion Matrix](<confusion_matrix_Test Data.png>)

> Metrics such as accuracy, precision, recall, and F1-score are tracked for each experiment run in MLflow, making it easy to compare model versions and roll back if needed.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/your-feature-name`
3. **Commit** your changes: `git commit -m "Add your feature"`
4. **Push** to the branch: `git push origin feature/your-feature-name`
5. **Open** a Pull Request

Please ensure your code follows the existing style and all tests pass before submitting a PR.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Made with ❤️ by <a href="https://github.com/DeepuML">DeepuML</a></p>
