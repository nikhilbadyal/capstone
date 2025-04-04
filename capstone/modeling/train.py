import numpy as np
from sklearn.linear_model import LogisticRegression

from capstone.config import PROCESSED_TRAIN_DATA_FILE
from capstone.environment import PARAMS_FILE
from capstone.logger import logging
from capstone.utils import load_data, load_params, save_model


def train_model(
    x_train: np.ndarray, y_train: np.ndarray, random_state: int
) -> LogisticRegression:
    """Train the Logistic Regression model."""
    try:
        clf = LogisticRegression(
            C=1, solver="liblinear", penalty="l1", random_state=random_state
        )
        clf.fit(x_train, y_train)
        logging.info("Model training completed")
        return clf
    except Exception as e:
        logging.error("Error during model training: %s", e)
        raise


def main():
    try:

        train_data = load_data(PROCESSED_TRAIN_DATA_FILE)
        x_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        params_file = PARAMS_FILE.get()
        params = load_params(params_path=params_file)
        random_state = params["model_training"]["random_state"]
        model_name = params["model_training"]["model_name"]

        clf = train_model(x_train, y_train, random_state)

        save_model(clf, model_name)
    except Exception as e:
        logging.error("Failed to complete the model building process: %s", e)
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
