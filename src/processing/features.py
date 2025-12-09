# Used for feature generation, filtering, and selection.

import logging

import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def prepare_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the target variable DX_GROUP.
    Original ABIDE: 1=Autism, 2=Control.
    Machine Learning Standard: 1=Positive (ASD), 0=Negative (Control).
    """
    logger.info("Preparing Target Variable...")
    target = config.TARGET_COLUMN

    if target not in df.columns:
        raise ValueError(f"Target column {target} not found in dataframe.")
    
    # Check current values
    unique_vals = df[target].unique()
    logger.info(f"Original Target values: {sorted(unique_vals)} (1=ASD, 2=Control)")
    logger.info(f"ASD Length = {len(df[df[target]==1])} | Control Length = {len(df[df[target]==2])}")

    # Apply mapping
    df[target] = df[target].map(config.TARGET_MAPPING)

    # Verify mapping
    new_vals = df[target].unique()
    logger.info(f"Mapped Target values: {sorted(new_vals)} (1=ASD, 0=Control)")
    logger.info(f"ASD Length = {len(df[df[target]==1])} | Control Length = {len(df[df[target]==0])}")

    # Drop rows with NaN targets (if any survived imputation)
    before_n = len(df)
    df = df.dropna(subset=[target])
    after_n = len(df)

    if before_n != after_n:
        logger.warning(f"Dropped {before_n - after_n} rows with missing Target.")

    return df
