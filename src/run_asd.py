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

def main():
    # Initial missing values report
    # logger.info(f"Initial missing value report shape: {initial_missing_values_report.shape}")
    # plot_missing_distribution(initial_missing_values_report)

    # Remaining missing values report
    # logger.info(f"Remaining missing value report shape: {remaining_missing_values_report.shape}")
    # plot_missing_distribution(remaining_missing_values_report)

    df_imputed = utils.load_imputed_data()
    logger.info(f"Loaded Imputed Data. Shape: {df_imputed.shape}")


if __name__ == '__main__':
    main()
