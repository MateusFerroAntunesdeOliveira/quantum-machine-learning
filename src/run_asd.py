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

from . import config
from . import utils
from . import logger
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

def impute_missing(df):
    """
    Advanced imputation for three groups of colunas:
      - CORE_ATTRIBUTES (all numeric): IterativeImputer with BayesianRidge MICE
      - SUPPORTING_ATTRIBUTES:
          - numeric: KNNImputer(n_neighbors=5)
          - categorical: SimpleImputer(strategy='most_frequent')
      - Default:
          - numeric: SimpleImputer(strategy='median')
          - categorical: SimpleImputer(strategy='most_frequent')
    """
    logger.info(f"Imputing missing values with advanced strategies...")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()

    # * Safe intersection - ensure columns exist in df
    core_num =    [c for c in config.CORE_ATTRIBUTES       if c in num_cols]
    support_num = [c for c in config.SUPPORTING_ATTRIBUTES if c in num_cols]
    support_cat = [c for c in config.SUPPORTING_ATTRIBUTES if c in cat_cols]

    # * What's left are default columns
    defined_cols = set(core_num + support_num + support_cat)
    default_num = [c for c in num_cols if c not in defined_cols]
    default_cat = [c for c in cat_cols if c not in defined_cols]

    logger.debug(f"Core Numeric columns ({len(core_num)}): {core_num}")
    logger.debug(f"Support Numeric columns ({len(support_num)}): {support_num}")
    logger.debug(f"Support Categorical columns ({len(support_cat)}): {support_cat}")
    logger.debug(f"Default Numeric columns ({len(default_num)}): {default_num}")
    logger.debug(f"Default Categorical columns ({len(default_cat)}): {default_cat}\n")

    # * Define imputers for each group
    core_numerical_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    support_numerical_imputer = KNNImputer(n_neighbors=5)
    simple_frequent_imputer = SimpleImputer(strategy='most_frequent')
    simple_median_imputer = SimpleImputer(strategy='median')

    # * Column Transformer with verbose_feature_names_out=False to keep original clean names
    transformer = ColumnTransformer([
        ('core_num',  core_numerical_imputer,    core_num),
        ('sup_num',   support_numerical_imputer, support_num),
        ('sup_cat',   simple_frequent_imputer,   support_cat),
        ('num_def',   simple_median_imputer,     default_num),
        ('cat_def',   simple_frequent_imputer,   default_cat),
    ], remainder='drop', verbose_feature_names_out=False)

    # * Enable direct pandas output - new in sklearn 1.2+
    transformer.set_output(transform='pandas')
    
    logger.info(f"Fitting and transforming data for imputation...")
    df_imputed = transformer.fit_transform(df)
    missing_after = df_imputed.isnull().sum().sum()
    logger.info(f"Successfully performed imputation. Total missing values after imputation: {missing_after}\n")

    return df_imputed

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
    raw_df = utils.load_raw_data()
    df = cleaning.initial_cleanup(raw_df)

    # * Initial missing values report
    initial_missing_values_report = utils.missing_value_report(df)
    logger.info(f"Initial missing value report shape: {initial_missing_values_report.shape}")
    plot_missing_distribution(initial_missing_values_report)

    # * Drop columns by threshold
    df_clean, dropped = cleaning.drop_columns_by_threshold(df=df, report=initial_missing_values_report)
    logger.info(f"Columns dropped: {len(dropped)}")
    logger.info(f"Columns remaining after drop: {df_clean.shape[1]}")

    # * Remaining missing values report
    remaining_missing_values_report = utils.missing_value_report(df_clean)
    logger.info(f"Remaining missing value report shape: {remaining_missing_values_report.shape}\n")
    plot_missing_distribution(remaining_missing_values_report)
    remaining_missing_values_report.to_csv(config.DROPPED_COLS_FILE)

    # * Impute missing values
    df_imputed = impute_missing(df_clean)

    # * Ensure TARGET_COLUMN (DX_GROUP) is integer after imputation    
    if config.TARGET_COLUMN in df_imputed.columns:
        df_imputed[config.TARGET_COLUMN] = df_imputed[config.TARGET_COLUMN].astype(int)  

    df_imputed.to_csv(config.IMPUTED_DATA_FILE, index=False)
    logger.info(f"Imputed data saved to: {config.IMPUTED_DATA_FILE}")

    pearson = compute_pearson_correlation(df_imputed)
    plt.figure(figsize=(12, 10))
    sns.heatmap(pearson, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Pearson Correlation Matrix')
    plt.show()

    spearman = compute_spearman_correlation(df_imputed)
    plt.figure(figsize=(12, 10))
    sns.heatmap(spearman, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Spearman Correlation Matrix')
    plt.show()

    pps_df = compute_pps_matrix(df_imputed, target_col=config.TARGET_COLUMN)
    n_feats = pps_df.shape[0]
    plt.figure(figsize=(12, max(4, n_feats * 0.3)))
    sns.heatmap(pps_df, annot=True, fmt='.2f', cmap='coolwarm', cbar_kws={'shrink': 0.5}, annot_kws={'size': 9}, linewidths=0.5, square=False)
    plt.yticks(ticks=np.arange(n_feats) + 0.5, labels=pps_df.index, rotation=0, fontsize=8)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.title(f'PPS predicting {config.TARGET_COLUMN}')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
