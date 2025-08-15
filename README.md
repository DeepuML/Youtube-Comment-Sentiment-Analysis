# Youtube-Comment-Sentiment-Analysis

A Chrome extension that detects the sentiment of YouTube comments in real-time using a trained machine learning model.  
The project is fully automated with **MLflow** for experiment tracking & model registry, **DVC** for data and model versioning, and a **CI/CD pipeline** for continuous training and deployment.

---

## 📌 Features

- **Data Pipeline (DVC)** – Handles raw YouTube comment data ingestion, preprocessing, and model training.
- **Sentiment Model** – Trained LightGBM model for positive/negative/neutral comment classification.
- **MLflow Integration** – Experiment tracking, model versioning, and registry.
- **Automated Model Registration** – Models are automatically pushed to MLflow Registry with stage/alias tagging.
- **Chrome Extension** – Displays comment sentiment directly on YouTube pages.
- **CI/CD Pipeline** – GitHub Actions workflow runs the full pipeline (`dvc repro`) and updates the deployed model.

---

## 📂 Project Organization



Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------


---

## 🚀 How It Works

1. **Data Ingestion**  
   - Fetches YouTube comment data (via API or local dataset)  
   - Stores raw data in `data/raw`  

2. **Preprocessing**  
   - Cleans text, removes noise, tokenizes  
   - Stores intermediate datasets in `data/interim`  

3. **Feature Engineering**  
   - Generates features for model training  
   - Saves processed data in `data/processed`  

4. **Model Training**  
   - Trains a LightGBM model on processed features  
   - Logs metrics and artifacts to MLflow  

5. **Model Evaluation**  
   - Generates metrics, confusion matrix, ROC curve  
   - Stores reports in `reports/`  

6. **Model Registration**  
   - Pushes the trained model to MLflow Model Registry  
   - Transitions to `Staging` or tags with `environment=staging`  

7. **Chrome Extension Integration**  
   - The extension loads the latest deployed model  
   - Highlights YouTube comments with sentiment color codes  

---

## ⚙️ Running the Project Locally

 1. Clone the repo
git clone https://github.com/<your-username>/youtube-comment-sentiment-analysis.git
cd youtube-comment-sentiment-analysis

 2. Create virtual environment
conda create -n yt_project python=3.10
conda activate yt_project

 3. Install dependencies
pip install -r requirements.txt
 
4. Reproduce pipeline
dvc repro

5. Push artifacts to remote storage
dvc push
