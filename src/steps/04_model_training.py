# * Final ML -> Results

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import logging

import pandas as pd

from sklearn.discriminant_analysis import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.shared import config, utils, logger
from src.processing import modeling

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 04: STARTING MODEL TRAINING ===\n")

    try:
        X, y = utils.load_final_data()
        logger.info(f"Final training data shape: X={X.shape} and y={y.shape}")

        # Flatten y to ensure it's a 1D array (sklearn requirement)
        if isinstance(y, pd.DataFrame):
            y = y.iloc[:, 0]

    except FileNotFoundError as e:
        logger.error(e)
        return

    # Define Models to Benchmarking
    # We use make_pipeline to include Scaling automatically inside CV folds
    # Probably=True is required for SVC to compute probabilities for ROC_AUC
    models_to_test = [
        (
            "Baseline (Dummy)", 
            DummyClassifier(strategy="most_frequent")
        ),
        (
            "SVM (Linear)", 
            make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42))
        ),
        (
            "SVM (RBF)", 
            make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
        )
    ]

    logger.info(f"Initialize {len(models_to_test)} models for benchmarking...\n")
    all_results = list()

    for name, model in models_to_test:
        # Calls the abstraction layer in modeling.py
        # This handles the Cross-Validate logic, metrics calculation, and logging
        model_results = modeling.train_and_evaluate(model_name=name, model=model, X=X, y=y, k_folds=10)





    logger.info("=== STEP 04: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
