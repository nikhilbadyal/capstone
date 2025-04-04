# data preprocessing

import os
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

from capstone.config import (
    DATA_DIR,
    INTERIM_DATA_DIR,
    TEST_DATA_FILE,
    TEST_PROCESSED_DATA_FILE,
    TRAIN_DATA_FILE,
    TRAIN_PROCESSED_DATA_FILE,
)
from capstone.logger import logging

nltk.download("wordnet")
nltk.download("stopwords")


def preprocess_dataframe(df, col="text"):
    """
    Preprocess a DataFrame by applying text preprocessing to a specific column.

    Args:
        df (pd.DataFrame): The DataFrame to preprocess.
        col (str): The name of the column containing text.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Initialize lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    def preprocess_text(text):
        """Helper function to preprocess a single text string."""
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        # Remove numbers
        text = "".join([char for char in text if not char.isdigit()])
        # Convert to lowercase
        text = text.lower()
        # Remove punctuations
        text = re.sub("[%s]" % re.escape(string.punctuation), " ", text)
        text = text.replace("؛", "")
        text = re.sub(r"\s+", " ", text).strip()
        # Remove stop words
        text = " ".join([word for word in text.split() if word not in stop_words])
        # Lemmatization
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    # Apply preprocessing to the specified column
    df[col] = df[col].apply(preprocess_text)

    # Remove small sentences (less than 3 words)
    # df[col] = df[col].apply(lambda x: np.nan if len(str(x).split()) < 3 else x)

    # Drop rows with NaN values
    df = df.dropna(subset=[col])
    logging.info("Data pre-processing completed")
    return df


def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv(TRAIN_DATA_FILE)
        test_data = pd.read_csv(TEST_DATA_FILE)

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, "review")
        test_processed_data = preprocess_dataframe(test_data, "review")

        # Store the data inside data/processed
        os.makedirs(INTERIM_DATA_DIR, exist_ok=True)

        train_processed_data.to_csv(TRAIN_PROCESSED_DATA_FILE, index=False)
        test_processed_data.to_csv(TEST_PROCESSED_DATA_FILE, index=False)

        logging.info("Processed data saved to %s", DATA_DIR)
    except Exception as e:
        logging.error("Failed to complete the data transformation process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
