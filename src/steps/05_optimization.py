# Optimization using Optuna - for hyperparameter tuning

import logging

import pandas as pd

from src.shared import utils, logger
from src.processing import tuning

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 05: STARTING OPTIMIZATION ===\n")

    try:
        X, y = utils.load_final_data()
        logger.info(f"Final training data shape: X={X.shape} and y={y.shape}")

        # Flatten y to ensure it's a 1D array (sklearn requirement)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

    except FileNotFoundError as e:
        logger.error(str(e))
        return

    # Define Models to Optimize
    # We select the TOP 2 from Step 04 based on previous benchmarking results
    models_to_optimize = ['LightGBM', 'XGBoost']
    logger.info(f"Models selected for optimization: {models_to_optimize}\n")

    for model_name in models_to_optimize:
        try:
            # Running 50 trials per model is usually enough for a good convergence
            # without taking hours on this dataset size.
            tuning.run_optimization(model_name, X, y, n_trials=50)

        except Exception as e:
            logger.error(f"Failed to optimize {model_name}: {str(e)}")

    logger.info("=== STEP 05: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
