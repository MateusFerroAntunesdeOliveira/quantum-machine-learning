# * Imputed -> Final ML Dataset

import logging

from src.shared import config, utils, logger
from src.processing import features

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 03: STARTING FEATURE ENGINEERING ===\n")

    # 4 sub-steps:
    
    # Preparation.
        # ABIDE uses 1 and 2 for groups. We need to change to 0 and 1 (binary).
    try:
        df_imputed = utils.load_imputed_data()
        logger.info(f"Loaded Imputed Data. Shape: {df_imputed.shape}\n")

        df_prepared = features.prepare_target_variable(df_imputed)
        logger.info(f"Prepared Target Variable. Shape: {df_prepared.shape}\n")

    except FileNotFoundError:
        logger.error("Imputed data not found! Please run 'Step 01' first.\n")
        return

    # Feature Generation. Maybe QI * Age...
        # But, we can not apply polynomial features to all of them. If we have 100 columns, we would have 1000+ features.
        # So, apply polynomial features (degree=2) only to CORE ATTRIBUTES (numeric) and some selected SUPPORTING ATTRIBUTES (numeric).
    df_expanded = features.generate_polynomial_features(df_prepared)

    checkpoint_path = config.OUTPUT_DIR / "data_checkpoint_poly.csv"
    df_expanded.to_csv(checkpoint_path, index=False)
    logger.info(f"Saved polynomial features checkpoint to {checkpoint_path}\n")

    # Feature Filtering.
        # Variance Thresholding to remove low-variance features.
        # Multicollinearity filter to remove highly correlated features. If Feature A and B have correlation > 0.95, drop one of them.

    # Feature Selection.
        # Reduce dataset to TOP-K features:
        # Mutual Information to capture non-linear relationships (between features and target).
        # Wrapper (RF Importance) to remove ignored features by model.
        # ! PCA destroy feature interpretability, so we will avoid it for now. As we wanna interpret the features later with SHAP.

    logger.info("=== STEP 03: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
