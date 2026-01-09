# * Final ML -> Results

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import logging

import pandas as pd

# Algorithms
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Pipeline & Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
        # Baseline
        (
            "Baseline (Dummy)", 
            DummyClassifier(strategy="most_frequent")
        ),
        # Distance-Based (requires scaling)
        (
            "SVM (Linear)", 
            make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42))
        ),
        (
            "SVM (RBF)", 
            make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42))
        ),
        (
            "KNN (k=5)", 
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
        ),
        (
            "KNN (k=10)", 
            make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10))
        ),
        # Ensemble / Tree-Based (no scaling needed)
        (
            "Random Forest", 
            RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        ),
        (
            "XGBoost", 
            XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, n_jobs=-1)
        ),
        (
            "LightGBM", 
            LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1)
        ),
        
    ]
    logger.info(f"Initialize {len(models_to_test)} models for benchmarking...\n")

    all_results = list()
    for name, model in models_to_test:
        # Calls the abstraction layer in modeling.py
        # This handles the Cross-Validate logic, metrics calculation, and logging
        model_results = modeling.train_and_evaluate(model_name=name, model=model, X=X, y=y, k_folds=10)

        if model_results:
            all_results.append(model_results)

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by='Mean_f1', ascending=False)

        utils.save_data(results_df, config.MODEL_COMPARISON_RESULTS_FILE, index_value=True, label="Model Comparison Results")
        logger.info("Final Benchmarking Summary (sorted by Mean F1 Score):")

        summary_cols = ['Model', 'Mean_f1', 'Mean_roc_auc', 'Mean_accuracy', 'Mean_recall']
        summary_df = results_df[summary_cols].copy()

        for index, row in summary_df.iterrows():
            logger.info(f"{row['Model']:<20} | F1: {row['Mean_f1']:.4f} | AUC: {row['Mean_roc_auc']:.4f}")

    modeling.plot_benchmark_results(results_df)

    logger.info("=== STEP 04: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
