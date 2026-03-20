import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Download required NLTK resources for text preprocessing
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_comment(comment):
    """Apply text cleaning and normalization to a single comment.

    Steps performed:
    - Lowercase and strip whitespace
    - Replace newlines with spaces
    - Remove non-alphanumeric characters except punctuation (! ? . ,)
    - Collapse multiple spaces
    - Remove stopwords while preserving key negation/contrast words
    - Lemmatize each token to its base form
    """
    try:
        # Normalize case and remove surrounding whitespace
        comment = comment.lower().strip()
        # Replace newline characters with a single space
        comment = re.sub(r'\n', ' ', comment)
        # Remove characters that are not letters, digits, spaces, or basic punctuation
        comment = re.sub(r'[^a-z0-9\s!?.,]', '', comment)
        # Collapse consecutive whitespace into a single space
        comment = re.sub(r'\s+', ' ', comment).strip()
        # Remove stopwords but keep sentiment-relevant negation/contrast words
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        tokens = [word for word in comment.split() if word not in stop_words]
        # Lemmatize each token to reduce inflectional forms
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    """Load the sentiment model from MLflow and the TF-IDF vectorizer from disk.

    Args:
        model_name (str): Registered model name in MLflow Model Registry.
        model_version (str): Specific model version to load.
        vectorizer_path (str): Local file path to the pickled TF-IDF vectorizer.

    Returns:
        tuple: (mlflow.pyfunc.PyFuncModel, TfidfVectorizer)
    """
    mlflow.set_tracking_uri("http://34.224.212.114:8000/")  # MLflow tracking server URI
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    # Load the model from the MLflow Model Registry
    model = mlflow.pyfunc.load_model(model_uri)
    # Load the TF-IDF vectorizer saved during training
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

# Load model and vectorizer once at startup to avoid reloading on every request
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "4", "./tfidf_vectorizer.pkl")

@app.route('/')
def home():
    """Health-check endpoint that confirms the API is running."""
    return "This is the Flask Api for the Youtube Comment Analysis..... "

@app.route('/predict_with_timestamps', methods=['POST'])
def predict_with_timestamps():
    """Predict sentiment for a list of comments that include timestamps.

    Expects JSON body:
        { "comments": [{"text": "...", "timestamp": "YYYY-MM-DD HH:MM:SS"}, ...] }

    Returns:
        JSON list of objects with 'comment', 'sentiment', and 'timestamp' fields.
    """
    data = request.json
    comments_data = data.get('comments')
    if not comments_data:
        return jsonify({"error": "No comments provided"}), 400
    try:
        # Separate comment text and timestamps from the input payload
        comments = [item['text'] for item in comments_data]
        timestamps = [item['timestamp'] for item in comments_data]
        # Preprocess and vectorize the comments
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        transformed_df = pd.DataFrame(transformed_comments, columns=vectorizer.get_feature_names_out())
        # Run inference and convert predictions to strings
        predictions = model.predict(transformed_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    # Build response pairing each comment with its sentiment and original timestamp
    response = [{"comment": comment, "sentiment": sentiment, "timestamp": timestamp}
                for comment, sentiment, timestamp in zip(comments, predictions, timestamps)]
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    """Predict sentiment for a list of plain text comments.

    Expects JSON body:
        { "comments": ["comment1", "comment2", ...] }

    Returns:
        JSON list of objects with 'comment' and 'sentiment' fields.
    """
    data = request.json
    comments = data.get('comments')
    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    try:
        # Preprocess and vectorize each comment
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments).toarray()
        transformed_df = pd.DataFrame(transformed_comments, columns=vectorizer.get_feature_names_out())
        # Run inference and convert predictions to strings
        predictions = model.predict(transformed_df).tolist()
        predictions = [str(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500
    # Pair each original comment with its predicted sentiment label
    response = [{"comment": comment, "sentiment": sentiment}
                for comment, sentiment in zip(comments, predictions)]
    return jsonify(response)

@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    """Generate a pie chart image from sentiment counts.

    Expects JSON body:
        { "sentiment_counts": {"1": <positive_count>, "0": <neutral_count>, "-1": <negative_count>} }

    Returns:
        PNG image of the sentiment distribution pie chart.
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
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        # Colors corresponding to Positive, Neutral, and Negative slices
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Ensure the pie is drawn as a circle
        # Render the chart to an in-memory buffer and return as PNG
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500

@app.route('/generate_wordcloud', methods=['POST'])
def generate_wordcloud():
    """Generate a word-cloud image from a list of comments.

    Expects JSON body:
        { "comments": ["comment1", "comment2", ...] }

    Returns:
        PNG image of the word cloud built from the preprocessed comment text.
    """
    try:
        data = request.get_json()
        comments = data.get('comments')
        if not comments:
            return jsonify({"error": "No comments provided"}), 400
        # Preprocess and join all comments into a single text corpus
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        text = ' '.join(preprocessed_comments)
        # Build word cloud with English stopwords removed
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)
        # Save the word cloud to an in-memory buffer and return as PNG
        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_wordcloud: {e}")
        return jsonify({"error": f"Word cloud generation failed: {str(e)}"}), 500

@app.route('/generate_trend_graph', methods=['POST'])
def generate_trend_graph():
    """Generate a monthly sentiment trend line graph from timestamped sentiment data.

    Expects JSON body:
        { "sentiment_data": [{"timestamp": "YYYY-MM-DD", "sentiment": <-1|0|1>}, ...] }

    Returns:
        PNG image showing the percentage of Positive, Neutral, and Negative comments
        per month over time.
    """
    try:
        data = request.get_json()
        sentiment_data = data.get('sentiment_data')
        if not sentiment_data:
            return jsonify({"error": "No sentiment data provided"}), 400
        # Build a DataFrame with a datetime index for time-series resampling
        df = pd.DataFrame(sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)
        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        # Resample to monthly frequency and count occurrences of each sentiment value
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        # Convert counts to percentages for a normalized view
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100
        # Ensure all three sentiment columns exist even if absent in the data
        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0
        monthly_percentages = monthly_percentages[[-1, 0, 1]]
        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}
        # Plot a separate line for each sentiment category
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
        # Format x-axis ticks as 'YYYY-MM' with automatic spacing
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()
        # Save the plot to an in-memory buffer and return as PNG
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()
        return send_file(img_io, mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Error in /generate_trend_graph: {e}")
        return jsonify({"error": f"Trend graph generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)
