# feature engineering

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from capstone.config import (
    INTERIM_TEST_DATA_FILE,
    INTERIM_TRAIN_DATA_FILE,
    PROCESSED_TEST_DATA_FILE,
    PROCESSED_TRAIN_DATA_FILE,
)
from capstone.environment import PARAMS_FILE
from capstone.logger import logging
from capstone.utils import load_data, load_params, save_data, save_model


def apply_bow(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    max_features: int,
    vectorizer_name: str,
) -> tuple:
    """Apply Count Vectorizer to the data."""
    try:
        logging.info("Applying BOW...")
        vectorizer = CountVectorizer(max_features=max_features)

        x_train = train_data["review"].values
        y_train = train_data["sentiment"].values
        x_test = test_data["review"].values
        y_test = test_data["sentiment"].values

        x_train_bow = vectorizer.fit_transform(x_train)
        x_test_bow = vectorizer.transform(x_test)

        train_df = pd.DataFrame(x_train_bow.toarray())
        train_df["label"] = y_train

        test_df = pd.DataFrame(x_test_bow.toarray())
        test_df["label"] = y_test
        save_model(vectorizer, vectorizer_name)
        logging.info("Bag of Words applied and data transformed")

        return train_df, test_df
    except Exception as e:
        logging.error("Error during Bag of Words transformation: %s", e)
        raise


def main():
    try:
        params_file = PARAMS_FILE.get()
        params = load_params(params_path=params_file)
        max_features = params["feature_engineering"]["max_features"]
        vectorizer_name = params["feature_engineering"]["vectorizer_name"]

        train_data = load_data(INTERIM_TRAIN_DATA_FILE)
        test_data = load_data(INTERIM_TEST_DATA_FILE)

        train_df, test_df = apply_bow(
            train_data, test_data, max_features, vectorizer_name
        )

        save_data(train_df, PROCESSED_TRAIN_DATA_FILE, keep_index=False)
        save_data(test_df, PROCESSED_TEST_DATA_FILE, keep_index=False)
    except Exception as e:
        logging.error("Failed to complete the feature engineering process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
