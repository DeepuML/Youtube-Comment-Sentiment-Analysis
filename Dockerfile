FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

COPY flask_app/ /app/

# Copy vectorizer if present at build time; it can also be mounted at runtime
COPY tfidf_vectorizer.pkl /app/tfidf_vectorizer.pkl

RUN pip install --no-cache-dir -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet omw-1.4

# Runtime configuration — override these via docker run -e or docker-compose environment
ENV MLFLOW_TRACKING_URI="http://localhost:5000"
ENV MODEL_NAME="yt_chrome_plugin_model"
ENV MODEL_VERSION="Production"
ENV VECTORIZER_PATH="/app/tfidf_vectorizer.pkl"

EXPOSE 5000

CMD ["python", "app.py"]