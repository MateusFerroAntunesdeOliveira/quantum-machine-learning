# Explainability Analysis Step using SHAP for model interpretability

import logging

import pandas as pd

from src.shared import config, utils, logger
from src.processing import explainability

logger.setup_logging()
logger = logging.getLogger(__name__)

CURRENT_BEST_MODEL = "LightGBM"

def main():
    logger.info("=== STEP 06: STARTING EXPLAINABILITY ANALYSIS ===\n")

    try:
        X, y = utils.load_final_data()
        logger.info(f"Final training data shape: X={X.shape} and y={y.shape}")

        # Flatten y to ensure it's a 1D array (sklearn requirement)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

    except FileNotFoundError as e:
        logger.error(str(e))
        return

    params_file = f"{config.BEST_PARAMS_FILE}_{CURRENT_BEST_MODEL}.json"
    if not utils.file_exists(params_file):
        logger.error(f"Best hyperparameters file not found: {params_file}")
        return

    # * Train the best model using the optimized hyperparameters
    try:
        model = explainability.train_best_model(X, y, params_file)
    except Exception as e:
        logger.error(f"Error training the best model: {e}")
        return

    # * Perform SHAP analysis for explainability
    try:
        explainability.perform_shap_analysis(model, X)
    except Exception as e:
        logger.error(f"Error during SHAP analysis: {e}")
        return

    logger.info("=== STEP 06: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
