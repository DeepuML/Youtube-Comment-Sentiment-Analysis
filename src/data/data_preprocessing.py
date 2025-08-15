# src/data/data_preprocessing.py

import numpy as np
import pandas as pd
import os
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging

# --------------------------------------------------
# Logging configuration for data preprocessing stage
# --------------------------------------------------
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Console handler for debug-level logs
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File handler to store error-level logs in a file
file_handler = logging.FileHandler('preprocessing_errors.log')
file_handler.setLevel('ERROR')

# Log formatting: timestamp - logger name - level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add both console and file handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# --------------------------------------------------
# Download required NLTK resources for lemmatization and stopwords
# --------------------------------------------------
nltk.download('wordnet')
nltk.download('stopwords')

# --------------------------------------------------
# Function: Preprocess individual comments
# --------------------------------------------------
def preprocess_comment(comment):
    """Apply text cleaning and normalization to a single comment."""
    try:
        # Convert text to lowercase for uniformity
        comment = comment.lower()

        # Remove leading and trailing spaces
        comment = comment.strip()

        # Replace newline characters with spaces
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters except for punctuation marks like ! ? . ,
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but keep key negation/contrast words for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize each word to its base form
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        # Log the error and return the original comment without changes
        logger.error(f"Error in preprocessing comment: {e}")
        return comment

# --------------------------------------------------
# Function: Apply preprocessing to entire dataframe
# --------------------------------------------------
def normalize_text(df):
    """Apply preprocessing to all rows in the dataframe's 'clean_comment' column."""
    try:
        # Apply preprocessing function to each comment
        df['clean_comment'] = df['clean_comment'].apply(preprocess_comment)
        logger.debug('Text normalization completed')
        return df
    except Exception as e:
        logger.error(f"Error during text normalization: {e}")
        raise

# --------------------------------------------------
# Function: Save processed datasets to interim directory
# --------------------------------------------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save preprocessed train and test datasets into the 'interim' folder."""
    try:
        # Create interim data folder inside provided data path
        interim_data_path = os.path.join(data_path, 'interim')
        logger.debug(f"Creating directory {interim_data_path}")
        
        os.makedirs(interim_data_path, exist_ok=True)  # Make sure folder exists
        logger.debug(f"Directory {interim_data_path} created or already exists")

        # Save processed training and testing datasets as CSV
        train_data.to_csv(os.path.join(interim_data_path, "train_processed.csv"), index=False)
        test_data.to_csv(os.path.join(interim_data_path, "test_processed.csv"), index=False)
        
        logger.debug(f"Processed data saved to {interim_data_path}")
    except Exception as e:
        logger.error(f"Error occurred while saving data: {e}")
        raise

# --------------------------------------------------
# Main function: Load raw data, preprocess, and save results
# --------------------------------------------------
def main():
    try:
        logger.debug("Starting data preprocessing...")
        
        # Load raw training and testing datasets from data/raw directory
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded successfully')

        # Apply preprocessing to train and test data
        train_processed_data = normalize_text(train_data)
        test_processed_data = normalize_text(test_data)

        # Save processed data into data/interim directory
        save_data(train_processed_data, test_processed_data, data_path='./data')
    except Exception as e:
        # Log and print error if preprocessing fails
        logger.error('Failed to complete the data preprocessing process: %s', e)
        print(f"Error: {e}")

# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == '__main__':
    main()
