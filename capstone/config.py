from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
TRAIN_DATA_FILE = RAW_DATA_DIR / "train.csv"
TEST_DATA_FILE = RAW_DATA_DIR / "test.csv"
INTERIM_DATA_DIR = DATA_DIR / "interim"
INTERIM_TRAIN_DATA_FILE = INTERIM_DATA_DIR / "train_processed.csv"
INTERIM_TEST_DATA_FILE = INTERIM_DATA_DIR / "test_processed.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_TRAIN_DATA_FILE = PROCESSED_DATA_DIR / "train_bow.csv"
PROCESSED_TEST_DATA_FILE = PROCESSED_DATA_DIR / "test_bow.csv"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
EXPERIMENT_INFO_PATH = REPORTS_DIR / "experiment_info.json"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
