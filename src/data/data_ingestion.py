# src/data/data_ingestion.py
#
# Data ingestion stage for the YouTube Comment Sentiment Analysis pipeline.
#
# This module is the first stage in the DVC-orchestrated ML pipeline.  It:
#   1. Loads raw Reddit comment data from a remote CSV source.
#   2. Removes missing values, duplicate rows, and empty comment strings.
#   3. Splits the cleaned dataset into training and test sets using the
#      split ratio defined in params.yaml.
#   4. Saves the resulting splits to data/raw/ for downstream pipeline stages.
#
# Usage (via DVC):
#   dvc repro data_ingestion
#
# Usage (standalone):
#   python src/data/data_ingestion.py

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import yaml
import logging

# --------------------------------------------------
# Logging configuration for data ingestion stage
# --------------------------------------------------
logger = logging.getLogger('data_ingestion')
logger.setLevel(logging.DEBUG)

# Console handler outputs all DEBUG and above messages to stdout
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# File handler captures only ERROR and above messages for post-run inspection
file_handler = logging.FileHandler('errors.log')
file_handler.setLevel(logging.ERROR)

# Consistent log format: timestamp - logger name - log level - message
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# --------------------------------------------------
# Function: Load pipeline parameters from YAML
# --------------------------------------------------
def load_params(params_path: str) -> dict:
    """Load pipeline configuration parameters from a YAML file.

    Args:
        params_path (str): Path to the params.yaml configuration file.

    Returns:
        dict: Parsed YAML content as a Python dictionary.

    Raises:
        FileNotFoundError: If the params file does not exist at the given path.
        yaml.YAMLError: If the file content is not valid YAML.
        Exception: Re-raises any other unexpected error.
    """
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# --------------------------------------------------
# Function: Load raw data from URL or local path
# --------------------------------------------------
def load_data(data_url: str) -> pd.DataFrame:
    """Load raw comment data from a CSV file path or URL.

    Args:
        data_url (str): Local file path or remote URL pointing to the CSV.

    Returns:
        pd.DataFrame: Loaded DataFrame containing the raw comment data.

    Raises:
        pd.errors.ParserError: If the CSV file cannot be parsed.
        Exception: Re-raises any other unexpected error.
    """
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data loaded from %s', data_url)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occurred while loading the data: %s', e)
        raise


# --------------------------------------------------
# Function: Clean the raw DataFrame
# --------------------------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Remove noise from the raw DataFrame before splitting.

    Performs three cleaning operations in order:
        1. Drop rows with any missing (NaN) values.
        2. Drop exact duplicate rows.
        3. Drop rows where the ``clean_comment`` column contains only
           whitespace (empty strings after stripping).

    Args:
        df (pd.DataFrame): Raw DataFrame loaded from the source CSV.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for train/test splitting.

    Raises:
        KeyError: If the expected ``clean_comment`` column is absent.
        Exception: Re-raises any other unexpected error.
    """
    try:
        # Remove rows that are entirely or partially missing
        df.dropna(inplace=True)

        # Remove exact duplicate rows to avoid data leakage
        df.drop_duplicates(inplace=True)

        # Remove rows where the comment field is blank after stripping whitespace
        df = df[df['clean_comment'].str.strip() != '']

        logger.debug('Data preprocessing completed: Missing values, duplicates, and empty strings removed.')
        return df
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error during preprocessing: %s', e)
        raise


# --------------------------------------------------
# Function: Persist train and test splits to disk
# --------------------------------------------------
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test DataFrames to the data/raw directory.

    Creates the ``data/raw`` directory if it does not already exist, then
    writes both DataFrames as CSV files without their index column.

    Args:
        train_data (pd.DataFrame): Training split to save.
        test_data (pd.DataFrame): Test split to save.
        data_path (str): Root data directory path (raw folder is created inside).

    Raises:
        Exception: Re-raises any unexpected error during directory creation
                   or file writing.
    """
    try:
        raw_data_path = os.path.join(data_path, 'raw')

        # Create the data/raw directory if it does not exist
        os.makedirs(raw_data_path, exist_ok=True)

        # Write both splits to CSV – index=False avoids writing the row numbers
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data: %s', e)
        raise


# --------------------------------------------------
# Main function: End-to-end data ingestion workflow
# --------------------------------------------------
def main():
    """Execute the full data ingestion workflow.

    Loads parameters, fetches raw data from the remote source, cleans it,
    splits it into training and test sets, and saves both sets to disk.
    """
    try:
        # Load the test split ratio from params.yaml located at the project root
        params = load_params(
            params_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '../../params.yaml'
            )
        )
        test_size = params['data_ingestion']['test_size']

        # Fetch the raw Reddit sentiment dataset from the public GitHub repository
        df = load_data(
            data_url='https://raw.githubusercontent.com/Himanshu-1703/reddit-sentiment-analysis/refs/heads/main/data/reddit.csv'
        )

        # Remove noise (missing values, duplicates, and empty comments)
        final_df = preprocess_data(df)

        # Split into stratified train / test sets using the configured ratio
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

        # Persist both splits to data/raw/ for the next pipeline stage
        save_data(
            train_data,
            test_data,
            data_path=os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '../../data'
            )
        )

    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == '__main__':
    main()