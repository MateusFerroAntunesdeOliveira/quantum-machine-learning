# * Imputed -> Final ML Dataset

import logging

from src.shared import config, utils, logger
from src.processing import features

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 03: STARTING FEATURE ENGINEERING ===\n")
    try:
        df_imputed = utils.load_imputed_data()
        logger.info(f"Data Loaded. Shape: {df_imputed.shape}\n")
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # * 1. Preparation.
    # ABIDE uses 1 and 2 for groups. We need to change to 0 and 1 (binary).
    try:
        X, y = features.prepare_target_variable(df_imputed)
        logger.info(f"Target Variable Prepared. Features shape: {X.shape}, Target shape: {y.shape}\n")
    except Exception as e:
        logger.error("Failed in Preparation step.", exc_info=e)
        return

    # * 1.5. Encode Categorical Features.
    try:
        X_encoded = features.encode_categorical_features(X)
        logger.info(f"Categorical Features Encoded. New shape: {X_encoded.shape}\n")
    except Exception as e:
        logger.error("Failed in Encoding Categorical Features step.", exc_info=e)
        return

    # * 2. Feature Generation.
    # Maybe QI * Age.
    # We can not apply polynomial features to all of them. If we have 100 columns, we would have 1000+ features.
    # So, apply polynomial features (degree=2) only to CORE ATTRIBUTES (numeric) and some selected SUPPORTING ATTRIBUTES (numeric).
    try:
        X_generated = features.generate_polynomial_features(X_encoded)
        logger.info(f"Polynomial Features Generated. New shape: {X_generated.shape}\n")
    except Exception as e:
        logger.error("Failed in Polynomial Feature Generation step.", exc_info=e)
        return

    # * 3. Feature Filtering.
    # Variance Thresholding to remove low-variance features.
    # Multicollinearity filter to remove highly correlated features. If Feature A and B have correlation > 0.95, drop one of them.
    try:
        X_filtered = features.apply_feature_filtering(X_generated)
        logger.info(f"Feature Filtering Applied. New shape: {X_filtered.shape}\n")
    except Exception as e:
        logger.error("Failed in Feature Filtering step.", exc_info=e)
        return

    # * 4. Feature Selection.
    # Reduce dataset to TOP-K features:
    # Mutual Information to capture non-linear relationships (between features and target).
    # Wrapper (RF Importance) to remove ignored features by model.
    # ! PCA destroy feature interpretability, so we will avoid it for now. As we wanna interpret the features later with SHAP.
    try:
        X_selected = features.apply_feature_selection(X_filtered, y)
        logger.info(f"Feature Selection Applied. New shape: {X_selected.shape}\n")
    except Exception as e:
        logger.error("Failed in Feature Selection step.", exc_info=e)
        return

    logger.info("=== STEP 03: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
