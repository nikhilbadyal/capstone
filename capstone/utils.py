import os
from pathlib import Path

import pandas as pd
import yaml

from capstone.logger import logging


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, "r") as file:
            params = yaml.safe_load(file)
        logging.debug("Parameters retrieved from %s", params_path)
        return params
    except FileNotFoundError:
        logging.error("File not found: %s", params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error: %s", e)
        raise


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        logging.info("Data loaded from %s", data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the data: %s", e)
        raise


def save_data(df: pd.DataFrame, file_path: Path, keep_index: bool = True) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=keep_index)
        logging.info("Data saved to %s", file_path)
    except Exception as e:
        logging.error("Unexpected error occurred while saving the data: %s", e)
        raise
