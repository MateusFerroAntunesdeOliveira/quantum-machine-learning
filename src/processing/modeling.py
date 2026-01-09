# Used for implementing Stratified K-Fold Cross-Validation in modeling and metric evaluation

import logging

import numpy as np
import pandas as pd

from time import time
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.shared import styles, translate, utils

# Get logger instance for this module
logger = logging.getLogger(__name__)

def train_and_evaluate(model_name: str, model: BaseEstimator, X: pd.DataFrame, y: pd.Series, k_folds: int = 10) -> dict:
    """
    Runs Stratified K-Fold Cross-Validation for a given model.
    This serves as the 'Outer Loop' of our validation strategy.

    Args:
        model_name: String identifier for logging (e.g., 'SVM').
        model: Sklearn-compatible estimator.
        X, y: Training data.
        k_folds: Number of folds (default 10).

    Returns:
        dict: Average scores and standard deviations.
    """
    logger.info(f'Training {model_name} with Stratified CV (k={k_folds})...')

    # Use StratifiedKFold to maintain class balance across folds
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    scoring = get_scoring_metrics()

    start_time = time()
    try:
        # n_jobs=-1 to utilize all CPU cores
        cv_results = cross_validate(
            estimator=model,
            X=X,
            y=y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        elapsed_time = time() - start_time
        logger.info(f"{model_name} training completed in {elapsed_time:.2f} seconds.")
    except Exception as e:
        logger.error(f"Error during cross-validation of {model_name}: {e}")
        return {}

    # Aggregate results
    results_summary = {
        'Model': model_name,
        'Time (s)': elapsed_time
    }

    # For each metric defined, compute mean and std
    for metric in scoring.keys():
        key_mean = f'test_{metric}'
        if key_mean in cv_results:
            mean_score = np.mean(cv_results[key_mean])
            std_score = np.std(cv_results[key_mean])

            # Save readable format
            results_summary[f'Mean_{metric}'] = mean_score
            results_summary[f'Std_{metric}'] = std_score

            # Log key metrics
            if metric in ['f1', 'roc_auc']:
                logger.info(f"{metric.upper()}: {mean_score:.4f} (+/- {std_score:.4f})")

    logger.info(f"Completed training and evaluation for {model_name}.\n")
    return results_summary

def get_scoring_metrics() -> dict:
    """
    Returns a dictionary of metrics for Scikit-Learn cross_validate.
    Focus: F1 (imbalanced), AUC (discrimination), Recall (sensitivity).
    """
    logger.info('Setting up scoring metrics for model evaluation...')

    scoring_metrics = {
        'accuracy': 'accuracy',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'recall': 'recall',
        'precision': 'precision'
    }

    logger.info(f"Scoring Metrics: {list(scoring_metrics.keys())}")
    return scoring_metrics

def plot_benchmark_results(results_df: pd.DataFrame):
    """
    Generates a horizontal bar chart comparing models by F1-Score.
    Includes error bars (std dev) and highlights the winner.
    """
    logger.info("Generating Benchmarking Bar Plot...")

    if results_df.empty:
        logger.warning("No results to plot.")
        return

    # Filter out Baseline models for better visualization
    df_plot = results_df[~results_df['Model'].str.contains('Baseline')].copy()
    if df_plot.empty: df_plot = results_df.copy()

    # Create a map directly based on the model name to define colors
    winner_model = df_plot.iloc[0]['Model']
    palette_map = {m: styles.STANDARD_BLUE_COLOR if m == winner_model else styles.STANDARD_GRAY_COLOR for m in df_plot['Model']}

    # Smart Zoom Calculation
    min_score = df_plot['Mean_f1'].min()
    xlim = (max(0, min_score - 0.03), 1.015)

    utils.plot_generic_barplot(
        data=df_plot,
        x='Mean_f1',
        y='Model',
        hue='Model',
        palette=palette_map,
        xerr='Std_f1',
        text_labels_col='Mean_f1',
        title=translate.PLOT_LABELS['benchmark_title'],
        xlabel=translate.PLOT_LABELS['benchmark_xlabel'],
        ylabel='',
        filename="04_benchmarking_barplot.png",
        xlim=xlim
    )
