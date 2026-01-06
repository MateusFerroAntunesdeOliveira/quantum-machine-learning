# * Utils for Load, Save, Reports, and common operations

import logging

import pandas as pd

from . import config
from . import logger

logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Loads raw data from the path defined in config."""
    if not config.RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"File not found: {config.RAW_DATA_FILE}")

    logger.info(f"Loading raw data from: {config.RAW_DATA_FILE}")
    return pd.read_csv(config.RAW_DATA_FILE)

def load_imputed_data() -> pd.DataFrame:
    """Loads the processed (imputed) dataset."""
    if not config.IMPUTED_DATA_FILE.exists():
        raise FileNotFoundError(f"Imputed file not found. Run step 01 first.")

    logger.info(f"Loading imputed data from: {config.IMPUTED_DATA_FILE}")
    return pd.read_csv(config.IMPUTED_DATA_FILE)

def load_final_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the final (X_train, y_train) dataset for modeling."""
    if not config.FINAL_TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Featured file not found. Run step 03 first.")

    if not config.FINAL_TARGET_DATA_FILE.exists():
        raise FileNotFoundError(f"Target file not found. Run step 03 first.")

    logger.info(f"Loading final training (X) and target (y) data")
    X = pd.read_csv(config.FINAL_TRAIN_DATA_FILE)
    y = pd.read_csv(config.FINAL_TARGET_DATA_FILE).squeeze()  # Convert to Series

    return X, y

def save_data(df: pd.DataFrame, path: str, index_value: str, label="data"):
    """Generic CSV saver."""
    logger.info(f"Saving {label} to: {path}")
    df.to_csv(path, index=index_value)

def file_exists(path: str) -> bool:
    """Checks if a file exists at the given path."""
    from pathlib import Path
    return Path(path).exists()

def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the percentage of missing values for columns that have missing data.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Column', config.PERCENT_MISSING_COLUMN] sorted by config.PERCENT_MISSING_COLUMN descending.
    """
    missing_value = df.isnull().sum()
    missing_value = missing_value.rename("Total Missing").to_frame()
    
    missing_value[config.PERCENT_MISSING_COLUMN] = 100 * missing_value['Total Missing'] / len(df)
    missing_value.drop("Total Missing", axis=1, inplace=True)

    return missing_value
