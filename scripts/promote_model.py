# promote model

import mlflow

from capstone.environment import MLFLOW_TRACKING_URI, PARAMS_FILE
from capstone.utils import load_params


def promote_model():
    # Set up DagsHub credentials for MLflow tracking
    params_file = PARAMS_FILE.get()
    params = load_params(params_path=params_file)
    model_name = params["model_training"]["model_name"]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI.get(not_exists_okay=False))

    client = mlflow.MlflowClient()

    # Get the latest version in staging
    latest_version_staging = client.get_latest_versions(model_name, stages=["Staging"])[
        0
    ].version

    # Archive the current production model
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    for version in prod_versions:
        client.transition_model_version_stage(
            name=model_name, version=version.version, stage="Archived"
        )

    # Promote the new model to production
    client.transition_model_version_stage(
        name=model_name, version=latest_version_staging, stage="Production"
    )
    print(f"Model version {latest_version_staging} promoted to Production")


if __name__ == "__main__":
    promote_model()
