# flask_app/app.py
#
# Flask REST API server for the YouTube Comment Sentiment Analysis project.
#
# This module exposes the trained LightGBM sentiment model via HTTP endpoints
# consumed by the companion Chrome extension.  It handles:
#   - Text preprocessing (identical pipeline to training time)
#   - Sentiment prediction (Positive / Neutral / Negative)
#   - Visualization generation (pie chart, word cloud, trend graph)
#
# Endpoints:
#   GET  /                          → Health check
#   POST /predict                   → Predict sentiment for a list of comments
#   POST /predict_with_timestamps   → Predict sentiment with timestamp metadata
#   POST /generate_chart            → Return a sentiment distribution pie chart (PNG)
#   POST /generate_wordcloud        → Return a word-cloud image for the given comments (PNG)
#   POST /generate_trend_graph      → Return a monthly sentiment trend graph (PNG)
#
# The model is loaded from MLflow Model Registry at startup.  The TF-IDF
# vectorizer is loaded from a local pickle file that was saved during training.

import matplotlib
matplotlib.use('Agg')  # Use the non-interactive Agg backend before importing pyplot to avoid display errors in headless server environments

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import numpy as np
import joblib
import nltk
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from mlflow.tracking import MlflowClient
import matplotlib.dates as mdates

# --------------------------------------------------
# Flask application setup
# --------------------------------------------------
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing so the Chrome extension can call the API

# --------------------------------------------------
# Download required NLTK resources at startup
# (wordnet for lemmatisation, stopwords for filtering)
# --------------------------------------------------
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


# --------------------------------------------------
# Helper: Text preprocessing
# --------------------------------------------------
def preprocess_comment(comment):
    """Clean and normalise a raw YouTube comment for model inference.

    Applies the same preprocessing pipeline that was used during model
    training so that the inference distribution matches the training
    distribution.

    Steps:
        1. Lowercase and strip leading/trailing whitespace.
        2. Replace newline characters with spaces.
        3. Remove non-alphanumeric characters (keep ! ? . ,).
        4. Collapse multiple whitespace characters into a single space.
        5. Remove English stopwords while preserving negation words
           (not, but, however, no, yet) that carry sentiment signal.
        6. Lemmatise each token to its base form.

    Args:
        comment (str): Raw comment text from YouTube.

    Returns:
        str: Cleaned and normalised comment text, or the original
             comment unchanged if an error occurs during preprocessing.
    """
    try:
        # Step 1: Lowercase and strip surrounding whitespace
        comment = comment.lower().strip()

        # Step 2: Replace newlines with spaces to keep the text as a single line
        comment = re.sub(r'\n', ' ', comment)

        # Step 3: Remove characters that are not alphanumeric or basic punctuation
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)

        # Step 4: Collapse consecutive whitespace into a single space
        comment = re.sub(r'\s+', ' ', comment).strip()

        # Step 5: Remove stopwords but retain key negation/contrast words
        #         that carry important sentiment information
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        tokens = [word for word in comment.split() if word not in stop_words]

        # Step 6: Lemmatise each token (e.g., "running" → "run")
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

        return " ".join(tokens)
    except Exception as e:
        # Return the original comment unchanged so the API can still respond
        print(f"Error in preprocessing comment: {e}")
        return comment


# --------------------------------------------------
# Helper: Load model and vectorizer from MLflow Registry
# --------------------------------------------------
def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load the trained sentiment model and TF-IDF vectorizer.

    Connects to the remote MLflow tracking server, retrieves the specified
    model version from the Model Registry, and loads the corresponding
    TF-IDF vectorizer from the local filesystem.

    Args:
        model_name (str): Registered model name in MLflow Model Registry.
        model_version (str): Version number of the model to load.
        vectorizer_path (str): Path to the pickled TF-IDF vectorizer file.

    Returns:
        tuple: A (model, vectorizer) pair where:
            - model: mlflow.pyfunc.PyFuncModel – loaded LightGBM model.
            - vectorizer: TfidfVectorizer – fitted TF-IDF transformer.
    """
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")
    client = MlflowClient()
    # Build the MLflow model URI using the registered model name and version
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


# --------------------------------------------------
# Load model and vectorizer at application startup
# --------------------------------------------------
# Version 4 of the registered model is used in production.
# The TF-IDF vectorizer must reside in the same directory as this script.
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "4", "./tfidf_vectorizer.pkl")


# --------------------------------------------------
# Route: Health check
# --------------------------------------------------
@app.route('/')
def home():
    """Health check endpoint.

    Returns:
        str: A simple status message confirming the API is running.
    """
    return "This is the Flask Api for the Youtube Comment Analysis..... "


# --------------------------------------------------
# Route: Predict sentiment with timestamps
# --------------------------------------------------
@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment for a list of comments that include timestamps.

    Expected JSON body::

        {
            "comments": [
                {"text": "Great video!", "timestamp": "2024-01-15T10:30:00Z"},
                ...
            ]
        }

    Returns:
        JSON list where each element contains the original comment text,
        its predicted sentiment label ("1" = Positive, "0" = Neutral,
        "-1" = Negative), and the original timestamp.

    Status codes:
        200 – Success
        400 – Missing or empty comments list
        500 – Internal preprocessing or model inference error
    """
    data = request.json
    comments_data = data.get('comments')

    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Separate comment text and timestamps from the request payload
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]

        # Apply the same text preprocessing used during model training
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Transform text to TF-IDF feature vectors and create a DataFrame
        # with the correct feature names expected by the model
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        transformed_df = pd.DataFrame(transformed_comments, columns=vectorizer.get_feature_names_out())

        # Run inference and convert numeric labels to strings for JSON serialisation
        predictions = model.predict(transformed_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Combine comments, predictions, and timestamps into the response
    response = [
        {"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
        for comment, sentiment, timestamp in zip(comments, predictions, timestamps)
    ]
    return jsonify(response)


# --------------------------------------------------
# Route: Predict sentiment (without timestamps)
# --------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a list of plain comment strings.

    Expected JSON body::

        {
            "comments": ["Great video!", "This is terrible", ...]
        }

    Returns:
        JSON list where each element contains the original comment text
        and its predicted sentiment label ("1" = Positive, "0" = Neutral,
        "-1" = Negative).

    Status codes:
        200 – Success
        400 – Missing or empty comments list
        500 – Internal preprocessing or model inference error
    """
    data = request.json
    comments = data.get('comments')

    if not comments:
        return jsonify({"error": "No comments provided"}), 400

    try:
        # Preprocess each comment to match the training-time text pipeline
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]

        # Vectorise and build a named DataFrame for the model
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        transformed_df = pd.DataFrame(transformed_comments, columns=vectorizer.get_feature_names_out())

        # Run model inference and stringify labels for JSON compatibility
        predictions = model.predict(transformed_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    # Build response pairing each original comment with its predicted sentiment
    response = [
        {"comment": comment, "sentiment": sentiment}
        for comment, sentiment in zip(comments, predictions)
    ]
    return jsonify(response)


# --------------------------------------------------
# Route: Generate sentiment distribution pie chart
# --------------------------------------------------
@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a sentiment distribution pie chart and return it as a PNG image.

    Expected JSON body::

        {
            "sentiment_counts": {"1": 120, "0": 45, "-1": 30}
        }

    The keys "1", "0", and "-1" correspond to Positive, Neutral, and
    Negative comment counts respectively.

    Returns:
        PNG image (mimetype: image/png) of the rendered pie chart.

    Status codes:
        200 – Success (PNG binary response)
        400 – Missing sentiment counts
        500 – Chart generation error
    """
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')

        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Map sentiment keys to human-readable labels
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]

        # Guard against an all-zero dataset which would cause a division error
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        # Brand colors: blue for Positive, gray for Neutral, red for Negative
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',   # Show percentage inside each slice
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Ensure the pie is drawn as a circle

        # Render the figure to an in-memory PNG buffer
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()  # Free memory held by the figure

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Route: Generate word cloud image
# --------------------------------------------------
@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate a word cloud image from the provided comments and return it as a PNG.

    The comments are preprocessed (stopwords removed, lemmatised) before
    the word cloud is built so that common filler words do not dominate
    the visualisation.

    Expected JSON body::

        {
            "comments": ["Great video!", "Amazing content", ...]
        }

    Returns:
        PNG image (mimetype: image/png) of the rendered word cloud.

    Status codes:
        200 – Success (PNG binary response)
        400 – Missing comments list
        500 – Word cloud generation error
    """
    try:
        data = request.get_json()
        comments = data.get('comments')

        if not comments:
            return jsonify({"error": "No comments provided"}), 400

        # Preprocess and join all comments into a single text blob
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)

        # Generate word cloud with a dark background and blue colour scheme
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),  # Additional stopword filtering
            collocations=False  # Disable bigram collocations to avoid redundant phrases
        ).generate(text)

        # Render to an in-memory buffer and return as PNG
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Route: Generate monthly sentiment trend graph
# --------------------------------------------------
@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate a monthly sentiment trend line graph and return it as a PNG.

    Groups the provided sentiment data by calendar month and computes the
    percentage of Positive, Neutral, and Negative comments for each month.
    The result is plotted as three overlapping line series.

    Expected JSON body::

        {
            "sentiment_data": [
                {"sentiment": "1",  "timestamp": "2024-01-15T10:30:00Z"},
                {"sentiment": "-1", "timestamp": "2024-02-03T08:00:00Z"},
                ...
            ]
        }

    Returns:
        PNG image (mimetype: image/png) of the rendered trend graph.

    Status codes:
        200 – Success (PNG binary response)
        400 – Missing sentiment data
        500 – Trend graph generation error
    """
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')

        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400

        # Build a DataFrame and parse timestamps for time-series resampling
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        # Human-readable labels for each numeric sentiment class
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}

        # Resample to monthly frequency and count occurrences of each sentiment value
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)

        # Convert raw counts to percentages of the monthly total
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        # Ensure all three sentiment columns exist (months may be missing some classes)
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        # Reorder columns to ensure consistent plotting order: Negative, Neutral, Positive
        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        # Plot one line per sentiment class
        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o',
                linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)

        # Format x-axis with readable month labels and automatic tick spacing
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))

        plt.legend()
        plt.tight_layout()

        # Write the figure to an in-memory buffer and return as PNG
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()  # Free memory held by the figure

        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500


# --------------------------------------------------
# Application entry point
# --------------------------------------------------
if __name__ == '__main__':
    # Run the development server on all interfaces at port 3000.
    # In production, the app is served by Gunicorn inside a Docker container.
    app.run(host='0.0.0.0', port=3000, debug=True)
