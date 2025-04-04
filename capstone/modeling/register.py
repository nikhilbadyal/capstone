# register model

import warnings

import mlflow

from capstone.config import EXPERIMENT_INFO_PATH
from capstone.environment import MLFLOW_TRACKING_URI, PARAMS_FILE
from capstone.logger import logging
from capstone.utils import load_model_info, load_params

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Set up MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.get(not_exists_okay=False))


def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"

        # Register the model
        model_version = mlflow.register_model(model_uri, model_name)

        # Transition the model to "Staging" stage
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )

        logging.debug(
            f"Model {model_name} version {model_version.version} registered and transitioned to Staging."
        )
    except Exception as e:
        logging.error("Error during model registration: %s", e)
        raise


def main():
    try:
        model_info = load_model_info(EXPERIMENT_INFO_PATH)

        params_file = PARAMS_FILE.get()
        params = load_params(params_path=params_file)
        model_name = params["model_training"]["model_name"]
        register_model(model_name, model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
