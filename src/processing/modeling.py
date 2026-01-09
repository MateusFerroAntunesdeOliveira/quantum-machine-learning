# Used for implementing Stratified K-Fold Cross-Validation in modeling and metric evaluation

import logging

import numpy as np
import pandas as pd

from time import time
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.shared import config, styles, translate, utils

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
    Generates horizontal bar charts for ALL metrics (F1, AUC, Accuracy, Recall, Precision).
    Iterates through metrics, sorts models by that metric, highlights the winner,
    and saves individual PNGs.
    """
    logger.info("Generating Benchmarking Plots for all metrics...")

    if results_df.empty:
        logger.warning("No results to plot.")
        return

    # Filter out Baseline models globally
    df_clean = results_df[~results_df['Model'].str.contains('Baseline')].copy()
    if df_clean.empty: 
        df_clean = results_df.copy()

    # Define metrics to plot
    metrics_config = [
        ('f1', 'Mean_f1', 'Std_f1'),
        ('roc_auc', 'Mean_roc_auc', 'Std_roc_auc'),
        ('accuracy', 'Mean_accuracy', 'Std_accuracy'),
        ('recall', 'Mean_recall', 'Std_recall'),
        ('precision', 'Mean_precision', 'Std_precision')
    ]

    for metric_name, mean_col, std_col in metrics_config:
        # Check if metric exists in DF
        if mean_col not in df_clean.columns:
            continue

        logger.info(f"-> Plotting {metric_name}...")

        # Sort by the current metric (Descending)
        # This ensures the "Best" for THIS metric is always on top
        df_plot = df_clean.sort_values(by=mean_col, ascending=False).copy()

        # Identify Winner for THIS metric
        winner_model = df_plot.iloc[0]['Model']

        # Define Palette (Winner = Blue, Others = Gray)
        palette_map = {
            m: styles.STANDARD_BLUE_COLOR if m == winner_model else styles.STANDARD_GRAY_COLOR 
            for m in df_plot['Model']
        }

        min_score = df_plot[mean_col].min()
        xlim_min = max(0, min_score - 0.03)
        xlim = (xlim_min, 1.015)
        title_key = f'benchmark_title_{metric_name}'
        xlabel_key = f'benchmark_xlabel_{metric_name}'

        # Safe get from translate
        title_text = translate.PLOT_LABELS.get(title_key, f'Comparação ({metric_name})')
        xlabel_text = translate.PLOT_LABELS.get(xlabel_key, f'Média {metric_name}')

        utils.plot_generic_barplot(
            data=df_plot,
            x=mean_col,
            y='Model',
            hue='Model',
            palette=palette_map,
            xerr=std_col,
            text_labels_col=mean_col,
            title=title_text,
            xlabel=xlabel_text,
            ylabel='',
            filename=config.MODEL_BENCHMARK_PLOT_TEMPLATE.format(metric_name),
            xlim=xlim
        )
