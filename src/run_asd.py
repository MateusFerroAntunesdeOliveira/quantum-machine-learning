import logging
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="ppscore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ppscore as pps

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer

from .shared import config
from .shared import utils
from .shared import logger
from .processing import cleaning

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

logger.setup_logging()
logger = logging.getLogger(__name__)

def plot_missing_distribution(missing_values, bins=20):
    """
    Plot histogram of % missing values across all features.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(missing_values[config.PERCENT_MISSING_COLUMN], bins=bins, color='blue', alpha=0.7)
    plt.title('Distribution of Missingness Across Features')
    plt.xlabel('% Missing Values')
    plt.ylabel('Number of Columns')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def compute_pearson_correlation(df):
    """
    Compute Pearson correlation for numeric variables, including target.
    """
    logger.info(f"Computing Pearson correlation...")
    num = df.select_dtypes(include=[np.number])
    pearson = num.corr(method='pearson')
    return pearson

def compute_spearman_correlation(df):
    """
    Compute Spearman correlation for numeric variables, including target.
    """
    logger.info(f"Computing Spearman correlation...")
    num = df.select_dtypes(include=[np.number])
    spearman = num.corr(method='spearman')
    return spearman

def compute_pps_matrix(df, target_col=config.TARGET_COLUMN):
    """
    Compute the Predictive Power Score (PPS) matrix for the given DataFrame.
    The target column is specified by the `target_col` parameter.
    """
    logger.info(f"Computing full PPS matrix...")
    pps_full_matrix = pps.matrix(df)
    sub = pps_full_matrix[(pps_full_matrix.y == target_col) & (pps_full_matrix.x != target_col)]
    pps_pivot = sub.pivot(index='x', columns='y', values='ppscore')
    return pps_pivot

def main():
    # Initial missing values report
    # logger.info(f"Initial missing value report shape: {initial_missing_values_report.shape}")
    # plot_missing_distribution(initial_missing_values_report)

    # Remaining missing values report
    # logger.info(f"Remaining missing value report shape: {remaining_missing_values_report.shape}")
    # plot_missing_distribution(remaining_missing_values_report)

    pass

    # pearson = compute_pearson_correlation(df_imputed)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(pearson, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    # plt.title('Pearson Correlation Matrix')
    # plt.show()

    # spearman = compute_spearman_correlation(df_imputed)
    # plt.figure(figsize=(12, 10))
    # sns.heatmap(spearman, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    # plt.title('Spearman Correlation Matrix')
    # plt.show()

    # pps_df = compute_pps_matrix(df_imputed, target_col=config.TARGET_COLUMN)
    # n_feats = pps_df.shape[0]
    # plt.figure(figsize=(12, max(4, n_feats * 0.3)))
    # sns.heatmap(pps_df, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.5}, annot_kws={'size': 9}, linewidths=0.5, square=False)
    # plt.yticks(ticks=np.arange(n_feats) + 0.5, labels=pps_df.index, rotation=0, fontsize=8)
    # plt.xticks(rotation=45, ha='right', fontsize=9)
    # plt.title(f'PPS predicting {config.TARGET_COLUMN}')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()
