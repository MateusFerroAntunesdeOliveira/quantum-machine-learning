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

def generate_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates degree-2 polynomial and interaction features for selected columns.
    Example: If inputs are [Age, ADOS], it creates [Age^2, ADOS^2, Age*ADOS].
    """
    logger.info("Generating Polynomial Features (Degree=2)...")

    # 1. Select candidates that actually exist in the dataframe
    poly_candidates = [cand for cand in config.POLYNOMIAL_ATTRIBUTES if cand in df.columns]

    if not poly_candidates:
        logger.warning("No polynomial candidates found in dataframe. Skipping generation.")
        return df

    logger.info(f"Selected {len(poly_candidates)} features for expansion: {poly_candidates}")

    # 2. Extract data for expansion
    X_poly_input = df[poly_candidates]

    # 3. Apply Transformation
    # include_bias=False prevents creating a column of 1s
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly_output = poly.fit_transform(X_poly_input)

    # 4. Create Feature Names
    # We use get_feature_names_out to get names like "AGE_AT_SCAN^2" or "AGE_AT_SCAN FIQ"
    new_feature_names = poly.get_feature_names_out(poly_candidates)
    
    # 5. Convert to DataFrame
    df_poly = pd.DataFrame(X_poly_output, columns=new_feature_names, index=df.index)

    # 6. Merge back
    # We only want to keep the NEW features (interactions/squares).
    # The original features (degree 1) are already in 'df', and poly returns them too.
    # So we filter columns that are NOT in the original df
    cols_to_add = [col for col in df_poly.columns if col not in df.columns]

    if not cols_to_add:
        logger.info("No new polynomial features were created (maybe they already existed?).")
        return df

    logger.info(f"Created {len(cols_to_add)} new polynomial features.")

    # Concatenate original df with new polynomial features
    df_final = pd.concat([df, df_poly[cols_to_add]], axis=1)

    logger.info(f"Shape after polynomial feature generation: {df_final.shape}")
    return df_final
