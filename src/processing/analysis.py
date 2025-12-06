# * Used for Pearson, Spearman and PPS (Predictive Power Score) calculations

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ppscore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ppscore as pps
import seaborn as sns

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def plot_missing_distribution(report_df: pd.DataFrame, fileName: str):
    """
    Generates a bar plot of missing values percentages and saves it.
    """
    logger.info("Generating missing values distribution plot...")

    if report_df.empty:
        logger.warning("No missing values to plot.\n")
        return

    plt.figure(figsize=(12, 6))
    col_name = config.PERCENT_MISSING_COLUMN
    if col_name in report_df.columns:
        plt.hist(report_df[col_name], bins=20, color='blue', alpha=0.7)
    else:
        plt.hist(report_df.iloc[:, 0], bins=20, color='blue', alpha=0.7)

    plt.title('Distribution of Missingness Across Features')
    plt.xlabel('% Missing Values')
    plt.ylabel('Number of Columns')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()

    output_path = config.OUTPUT_DIR / fileName
    plt.savefig(output_path)
    plt.close()

    logger.info(f"Plot saved to: {output_path}\n")

def compute_and_plot_correlations(df: pd.DataFrame):
    """
    Computes Pearson and Spearman correlations for numeric columns.
    Saves the matrices as CSV and heatmaps as PNG.
    """
    logger.info("Computing correlations (Pearson & Spearman)...")

    # Filter numeric columns only
    df_num = df.select_dtypes(include=[np.number])

    if df_num.empty:
        logger.warning("No numeric data found for correlation analysis.\n")
        return

    methods = ['pearson', 'spearman']

    for method in methods:
        logger.info(f"-> Calculating {method.capitalize()} correlation...")
        corr_matrix = df_num.corr(method=method)

        # Save CSV
        csv_path = config.OUTPUT_DIR / f"02_correlation_matrix_{method}.csv"
        corr_matrix.to_csv(csv_path)

        # Save Heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()

        plot_path = config.OUTPUT_DIR / f"02_heatmap_{method}.png"
        plt.savefig(plot_path)
        plt.close()

        logger.info(f"Saved {method} matrix and heatmap.\n")

def compute_pps_matrix(df: pd.DataFrame):
    """
    Computes the Predictive Power Score (PPS) matrix.
    Generates two visualizations:
    1. Full Matrix (All features vs All features)
    2. Target Predictors (All features predicting the Target Column)
    """
    logger.info("Computing PPS Matrix (Predictive Power Score)...")

    try:
        # Calculate full PPS matrix
        matrix_df = pps.matrix(df)[['x', 'y', 'ppscore', 'case', 'is_valid_score']]

        # * Visualization 1: Full PPS Matrix (Features vs Features)
        logger.info("-> Generating Full PPS Matrix")
        pps_pivot = matrix_df.pivot(columns='x', index='y', values='ppscore')

        csv_path = config.OUTPUT_DIR / "02_pps_matrix_raw.csv"
        matrix_df.to_csv(csv_path, index=False)

        pivot_path = config.OUTPUT_DIR / "02_pps_matrix_pivot.csv"
        pps_pivot.to_csv(pivot_path)

        plt.figure(figsize=(14, 12))
        sns.heatmap(pps_pivot, annot=False, cmap='rocket_r', linewidths=0.5)
        plt.title('Predictive Power Score (PPS) Matrix - Full')
        plt.tight_layout()

        plot_path = config.OUTPUT_DIR / "02_heatmap_pps_full.png"
        plt.savefig(plot_path)
        plt.close()
        
        logger.info(f"Saved Full PPS matrix and heatmap.\n")

        # * Visualization 2: Target Predictors (Features vs Target)
        target = config.TARGET_COLUMN
        logger.info(f"-> Generating PPS for Target Column: {target}")

        target_predictors = matrix_df[
            (matrix_df['y'] == target) & 
            (matrix_df['x'] != target)
        ].copy()

        # Sort descending by predictive power
        target_predictors = target_predictors.sort_values('ppscore', ascending=False)

        if not target_predictors.empty:
            # Pivot for heatmap (Index=Feature, Column=Target)
            pivot_target = target_predictors.set_index('x')[['ppscore']]

            n_feats = len(pivot_target)
            figsize_height = max(6, n_feats * 0.25)

            plt.figure(figsize=(8, figsize_height))
            sns.heatmap(pivot_target, annot=True, fmt='.2f', cmap='rocket_r', cbar_kws={'label': 'Predictive Power Score (PPS)'}, linewidths=0.5, annot_kws={'size': 9})
            plt.title(f'PPS: Features Predicting {target} (Descending)')
            plt.ylabel('Feature')
            plt.xlabel('Target')
            plt.tight_layout()

            plot_path_target = config.OUTPUT_DIR / "02_heatmap_pps_target_only.png"
            plt.savefig(plot_path_target)
            plt.close()

            logger.info(f"Saved Target-Specific PPS heatmap.\n")
        else:
            logger.warning(f"No predictors found for target {target} in PPS matrix.\n")

    except Exception as e:
        logger.error(f"Failed to compute PPS: {e}\n")
