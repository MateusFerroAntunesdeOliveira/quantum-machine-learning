# Used for feature generation, filtering, and selection.

import json
import logging

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.preprocessing import PolynomialFeatures

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def prepare_target_variable(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Standardizes the target variable DX_GROUP.
    Maps: 1 (Autism) -> 1, 2 (Control) -> 0.
    Returns X (features) and y (target) separated.
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

    y = df[target].astype(int)
    X = df.drop(columns=[target])
    return X, y

def encode_categorical_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Converts categorical columns (strings) to numeric using One-Hot Encoding.
    Handles specific cases like removing high-cardinality IDs.
    """
    logger.info("Encoding Categorical Features...")

    # ! Explicitly drop FILE_ID if present (It's a unique identifier, not a feature)
    # If we encode it, we get 1000+ columns
    if 'FILE_ID' in X.columns:
        logger.info("Dropping 'FILE_ID' before encoding (High Cardinality Identifier).")
        X = X.drop(columns=['FILE_ID'])

    # Identify object/category columns
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        logger.info("No categorical columns found to encode.\n")
        return X

    logger.info(f"Encoding {len(cat_cols)} columns: {cat_cols}")

    # * Apply One-Hot Encoding
    # drop_first=True avoids the dummy variable trap (multicollinearity)
    # dtype=int forces 0/1 instead of True/False
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True, dtype=int)
    return X_encoded

def generate_polynomial_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Generates degree-2 polynomial and interaction features for selected columns.
    Example: If inputs are [Age, ADOS], it creates [Age^2, ADOS^2, Age*ADOS].
    """
    logger.info("Generating Polynomial Features (Degree=2)...")

    # 1. Select candidates that actually exist in the dataframe
    poly_candidates = [cand for cand in config.POLYNOMIAL_ATTRIBUTES if cand in X.columns]

    if not poly_candidates:
        logger.warning("No polynomial candidates found in dataframe. Skipping generation.\n")
        return X

    logger.info(f"Selected {len(poly_candidates)} features for expansion: {poly_candidates}")

    # 2. Extract data for expansion
    X_poly_input = X[poly_candidates]

    # 3. Apply Transformation
    # include_bias=False prevents creating a column of 1s
    poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
    X_poly_output = poly.fit_transform(X_poly_input)

    # 4. Create Feature Names
    # We use get_feature_names_out to get names like "AGE_AT_SCAN^2" or "AGE_AT_SCAN FIQ"
    new_feature_names = poly.get_feature_names_out(poly_candidates)
    
    # 5. Convert to DataFrame
    df_poly = pd.DataFrame(X_poly_output, columns=new_feature_names, index=X.index)

    # 6. Merge back
    # We only want to keep the NEW features (interactions/squares).
    # The original features (degree 1) are already in 'df', and poly returns them too.
    # So we filter columns that are NOT in the original df
    cols_to_add = [col for col in df_poly.columns if col not in X.columns]

    if not cols_to_add:
        logger.info("No new polynomial features were created (maybe they already existed?).\n")
        return X

    logger.info(f"Created {len(cols_to_add)} new polynomial features.")

    # Concatenate original df with new polynomial features
    df_final = pd.concat([X, df_poly[cols_to_add]], axis=1)
    return df_final

def apply_feature_filtering(X: pd.DataFrame) -> pd.DataFrame:
    """
    Applies Variance Thresholding and Multicollinearity Filtering.
    A. Drops features with 0 variance (or very low defined in config).
    B. Drops features with correlation > threshold (keeps the first one).
    """
    logger.info("Applying Feature Filtering...")

    # * A. Variance Threshold
    # Remove Low-Variance Features and apply Fit to find columns to keep
    selector = VarianceThreshold(threshold=config.VARIANCE_THRESHOLD)
    selector.fit(X)

    # Get all features that are NOT constant (i.e., passed the variance threshold)
    X_var = X.loc[:, selector.get_support()]

    dropped_var_count = len(X.columns) - len(X_var.columns)
    if dropped_var_count > 0:
        logger.info(f"Dropped {dropped_var_count} low-variance features.")

    X = X_var

    # * B. Multicollinearity Filter
    # Calculates correlation matrix and drops one of the pairs if corr > threshold
    # Note: We use .abs() because strong negative correlation (-0.99) is also bad for multicollinearity
    corr_matrix = X.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > config.CORRELATION_THRESHOLD)]

    if to_drop:
        logger.info(f"Dropping {len(to_drop)} features due to multicollinearity: {to_drop}")
        X = X.drop(columns=to_drop)
    else:
        logger.info("No features dropped due to multicollinearity.")

    return X

def apply_feature_selection(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Applies feature selection methods to reduce the dataset to the most relevant features.
    Methods can include Mutual Information and Wrapper methods (e.g., Random Forest Importance).
    Note: PCA is avoided here to maintain feature interpretability.
    """
    logger.info("Applying Feature Selection...")

    # We use a Random Forest to judge feature importance
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    # SelectFromModel will pick features whose importance is greater than the mean
    selector = SelectFromModel(classifier, threshold='mean')
    selector.fit(X, y)

    # Get the mask of selected features, then filter the dataframe
    selected_mask = selector.get_support()
    selected_feats = X.columns[selected_mask].tolist()
    X_selected = X.loc[:, selected_mask]

    logger.info(f"Wrapper - Selected {len(selected_feats)} features out of {X.shape[1]}.")
    logger.info(f"Final Shape: {X_selected.shape}")

    # Save selected features list
    try:
        with open(config.SELECTED_FEATURES_FILE, 'w') as f:
            json.dump(selected_feats, f, indent=4)
        logger.info(f"Saved selected features list to: {config.SELECTED_FEATURES_FILE}")
    except Exception as e:
        logger.error(f"Failed to save feature list json: {e}")
    
    return X_selected
