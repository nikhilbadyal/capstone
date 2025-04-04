import json
from pathlib import Path
from typing import Any

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from capstone.config import (
    EXPERIMENT_INFO_PATH,
    METRICS_PATH,
    PROCESSED_TEST_DATA_FILE,
)
from capstone.environment import MLFLOW_TRACKING_URI, PARAMS_FILE
from capstone.logger import logging
from capstone.utils import load_data, load_model, load_params

# Set up MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.get(not_exists_okay=False))


def evaluate_model(clf, x_test: np.ndarray, y_test: np.ndarray) -> tuple[dict, Any]:
    """Evaluate the model and return the evaluation metrics."""
    try:
        y_pred = clf.predict(x_test)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "auc": auc,
        }
        logging.info("Model evaluation metrics calculated")
        return (
            metrics_dict,
            infer_signature(x_test, y_pred),
        )
    except Exception as e:
        logging.error("Error during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, file_path: str | Path) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump(metrics, file, indent=4)
        logging.info("Metrics saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the metrics: %s", e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {"run_id": run_id, "model_path": model_path}
        with open(file_path, "w") as file:
            # noinspection PyTypeChecker
            json.dump(model_info, file, indent=4)
        logging.debug("Model info saved to %s", file_path)
    except Exception as e:
        logging.error("Error occurred while saving the model info: %s", e)
        raise


def main():
    params_file = PARAMS_FILE.get()
    params = load_params(params_path=params_file)
    mlflow.set_experiment(params["model_evaluation"]["experiment_name"])
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            logging.debug(f"Logging into {mlflow.get_artifact_uri()}")
            params_file = PARAMS_FILE.get()
            params = load_params(params_path=params_file)
            model_name = params["model_training"]["model_name"]
            clf = load_model(model_name)
            test_data = load_data(PROCESSED_TEST_DATA_FILE)

            x_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics, signature = evaluate_model(clf, x_test, y_test)

            save_metrics(metrics, METRICS_PATH)

            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            # Log model parameters to MLflow
            if hasattr(clf, "get_params"):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            # Log model to MLflow
            mlflow.sklearn.log_model(
                clf, "model", input_example=x_test[0:1], signature=signature
            )

            # Save model info
            save_model_info(run.info.run_id, "model", EXPERIMENT_INFO_PATH)

            # Log the metrics file to MLflow
            mlflow.log_artifact(METRICS_PATH)

        except Exception as e:
            logging.exception("Failed to complete the model evaluation process: %s", e)
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
