# data ingestion
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from capstone.config import RAW_DATA_DIR
from capstone.data.connections.s3 import S3Operations
from capstone.environment import PARAMS_FILE, S3_ACCESS_KEY, S3_BUCKET, S3_SECRET_KEY
from capstone.logger import logging
from capstone.utils import load_params, save_data

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
        save_data(train_data, Path(RAW_DATA_DIR, "train.csv"), keep_index=False)
        save_data(test_data, Path(RAW_DATA_DIR, "test.csv"), keep_index=False)
    except Exception as e:
        logging.error("Failed to complete the data ingestion process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
