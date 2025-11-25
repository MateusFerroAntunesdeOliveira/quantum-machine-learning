# * Used for MICE and KNN imputation methods

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import logging

import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.compose import ColumnTransformer

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def run_advanced_imputation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies hybrid imputation strategy:
    - MICE (BayesianRidge) for Core Attributes
    - KNN for Supporting Numeric
    - Mode/Median for others
    - Passthrough for ID and Target (no changes)

    Returns:
        pd.DataFrame: Fully imputed dataset.
    """
    logger.info('Starting Advanced Imputation Strategy...')

    # Separate by data type
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # * Safe intersection - ensure columns exist in df
    core_num =    [c for c in config.CORE_ATTRIBUTES       if c in num_cols]
    support_num = [c for c in config.SUPPORTING_ATTRIBUTES if c in num_cols]
    support_cat = [c for c in config.SUPPORTING_ATTRIBUTES if c in cat_cols]

    # * What's left are default columns
    covered_cols = set(core_num + support_num + support_cat)
    default_num = [c for c in num_cols if c not in covered_cols]
    default_cat = [c for c in cat_cols if c not in covered_cols]

    logger.info(f"  Strategy Mapping:")
    logger.info(f"  - MICE (Core Num): {len(core_num)} cols")
    logger.info(f"  - KNN (Support Num): {len(support_num)} cols")
    logger.info(f"  - SimpleImputer Most Frequent (Support Cat): {len(support_cat)} cols")
    logger.info(f"  - SimpleImputer Median (Default Num): {len(default_num)} cols")
    logger.info(f"  - SimpleImputer Most Frequent (Default Cat): {len(default_cat)} cols")

    # * Define transformers for ColumnTransformer pipeline
    core_numerical_imputer = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
    support_numerical_imputer = KNNImputer(n_neighbors=5)
    simple_frequent_imputer = SimpleImputer(strategy='most_frequent')
    simple_median_imputer = SimpleImputer(strategy='median')

    transformers = [
        ('core_mice',   core_numerical_imputer,     core_num),
        ('sup_knn',     support_numerical_imputer,  support_num),
        ('sup_cat',     simple_frequent_imputer,    support_cat),
        ('num_def',     simple_median_imputer,      default_num),
        ('cat_def',     simple_frequent_imputer,    default_cat),
    ]
    # * Pipeline Execution, using verbose_feature_names_out=False to keep original clean names
    preprocessor = ColumnTransformer(transformers, remainder='drop', verbose_feature_names_out=False)

    # * Enable direct pandas output - new in sklearn 1.2+
    preprocessor.set_output(transform='pandas')

    logger.info("Fitting and transforming data...")
    try:
        df_imputed = preprocessor.fit_transform(df)
    except Exception as e:
        logger.error(f"Error during imputation: {e}")
        raise e

    # * Final adjustments after imputation, ensuring data integrity
    if config.TARGET_COLUMN in df_imputed.columns:
        df_imputed[config.TARGET_COLUMN] = df_imputed[config.TARGET_COLUMN].astype(int)

    if config.ID_COLUMN in df_imputed.columns:
        df_imputed[config.ID_COLUMN] = df_imputed[config.ID_COLUMN].astype(int)

    missing_after = df_imputed.isnull().sum().sum()
    logger.info(f"Imputation complete. Total missing values after imputation: {missing_after}")
    logger.info(f"Advanced Imputation Strategy finished successfully.\n")
    return df_imputed
