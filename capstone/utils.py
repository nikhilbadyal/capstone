import json
import os
from pathlib import Path

import joblib
import pandas as pd
import yaml

from capstone.config import MODELS_DIR
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


def load_data(data_url: str | Path) -> pd.DataFrame:
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


def get_model_path():
    """Get the path to the model directory."""
    try:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        logging.info("Model directory is ready at %s", MODELS_DIR)
        return MODELS_DIR
    except Exception as e:
        logging.error("Error occurred while creating the model directory: %s", e)
        raise


def save_model(model, model_name: str) -> None:
    """Save the trained model to a file."""
    try:

        model_path = get_model_path() / f"{model_name}.pkl"
        joblib.dump(model, model_path)
        logging.info("Model saved to %s", model_path)
    except Exception as e:
        logging.error("Error occurred while saving the model: %s", e)
        raise


def load_model(model_name: str | Path):
    """Load the trained model from a file."""
    model_path = get_model_path() / f"{model_name}.pkl"
    try:
        model = joblib.load(model_path)
        logging.info("Model loaded from %s", model_path)
        return model
    except FileNotFoundError:
        logging.error("File not found: %s", model_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model: %s", e)
        raise


def load_model_info(file_path: str | Path) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, "r") as file:
            model_info = json.load(file)
        logging.debug("Model info loaded from %s", file_path)
        return model_info
    except FileNotFoundError:
        logging.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logging.error("Unexpected error occurred while loading the model info: %s", e)
        raise
