# data ingestion
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from capstone.config import RAW_DATA_DIR
from capstone.data.connections.s3 import S3Operations
from capstone.environment import PARAMS_FILE, S3_ACCESS_KEY, S3_BUCKET, S3_SECRET_KEY
from capstone.logger import logging
from capstone.utils import load_params

pd.set_option("future.no_silent_downcasting", True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the data."""
    try:
        logging.info("pre-processing...")
        final_df = df[df["sentiment"].isin(["positive", "negative"])]
        final_df["sentiment"] = final_df["sentiment"].replace(
            {"positive": 1, "negative": 0}
        )
        logging.info("Data preprocessing completed")
        return final_df
    except KeyError as e:
        logging.error("Missing column in the dataframe: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error during preprocessing: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Save the train and test datasets."""
    try:
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        train_data.to_csv(os.path.join(RAW_DATA_DIR, "train.csv"), index=False)
        test_data.to_csv(os.path.join(RAW_DATA_DIR, "test.csv"), index=False)
        logging.debug("Train and test data saved to %s", RAW_DATA_DIR)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise


def main():
    try:
        params_file = PARAMS_FILE.get()
        params = load_params(params_path=params_file)
        test_size = params["data_ingestion"]["test_size"]

        s3 = S3Operations(
            S3_BUCKET.get(not_exists_okay=False),
            S3_ACCESS_KEY.get(not_exists_okay=False),
            S3_SECRET_KEY.get(not_exists_okay=False),
        )
        df = s3.fetch_file_from_s3(params["data_ingestion"]["raw_file"])

        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(
            final_df, test_size=test_size, random_state=42
        )
        save_data(train_data, test_data)
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
