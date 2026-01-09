# * Utils for Load, Save, Reports, and common operations

import logging

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from . import config
from . import logger

logger = logging.getLogger(__name__)

def load_raw_data() -> pd.DataFrame:
    """Loads raw data from the path defined in config."""
    if not config.RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"File not found: {config.RAW_DATA_FILE}")

    logger.info(f"Loading raw data from: {config.RAW_DATA_FILE}")
    return pd.read_csv(config.RAW_DATA_FILE)

def load_imputed_data() -> pd.DataFrame:
    """Loads the processed (imputed) dataset."""
    if not config.IMPUTED_DATA_FILE.exists():
        raise FileNotFoundError(f"Imputed file not found. Run step 01 first.")

    logger.info(f"Loading imputed data from: {config.IMPUTED_DATA_FILE}")
    return pd.read_csv(config.IMPUTED_DATA_FILE)

def load_final_data() -> tuple[pd.DataFrame, pd.Series]:
    """Loads the final (X_train, y_train) dataset for modeling."""
    if not config.FINAL_TRAIN_DATA_FILE.exists():
        raise FileNotFoundError(f"Featured file not found. Run step 03 first.")

    if not config.FINAL_TARGET_DATA_FILE.exists():
        raise FileNotFoundError(f"Target file not found. Run step 03 first.")

    logger.info(f"Loading final training (X) and target (y) data")
    X = pd.read_csv(config.FINAL_TRAIN_DATA_FILE)
    y = pd.read_csv(config.FINAL_TARGET_DATA_FILE).squeeze()  # Convert to Series

    return X, y

def save_data(df: pd.DataFrame, path: str, index_value: str, label="data"):
    """Generic CSV saver."""
    logger.info(f"Saved {label} to: {path}")
    df.to_csv(path, index=index_value)

def file_exists(path: str) -> bool:
    """Checks if a file exists at the given path."""
    from pathlib import Path
    return Path(path).exists()

def get_missing_value_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the percentage of missing values for columns that have missing data.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Column', config.PERCENT_MISSING_COLUMN] sorted by config.PERCENT_MISSING_COLUMN descending.
    """
    missing_value = df.isnull().sum()
    missing_value = missing_value.rename("Total Missing").to_frame()
    
    missing_value[config.PERCENT_MISSING_COLUMN] = 100 * missing_value['Total Missing'] / len(df)
    missing_value.drop("Total Missing", axis=1, inplace=True)

    return missing_value

def apply_plot_style():
    """
    Applies the standardized plot style defined in config.
    Should be called before generating any figure.
    """
    sns.set_style("whitegrid")
    plt.rcParams.update(config.VIZ_PARAMS)
    logger.info("Applied standardized plot style from config.")

def save_plot(filename: str):
    """
    Generic function to save the current matplotlib figure using config paths.
    """
    output_path = config.OUTPUT_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=config.VIZ_PARAMS['figure.dpi'])
    plt.close()
    logger.info(f"Plot saved to: {output_path}\n")

def plot_generic_heatmap(data: pd.DataFrame, title: str, filename: str, cmap: str, mask=None, annot=False, cbar_label=None):
    """
    Generic function to plot standardized heatmaps (Corr or PPS).
    """
    plt.figure(figsize=(12, 10))
    is_diverging = cmap == 'RdBu'

    sns.heatmap(
        data,
        mask=mask,
        annot=annot,
        cmap=cmap,
        vmax=1,
        vmin=-1 if is_diverging else 0,         # If PPS (Blues), start at 0
        center=0 if is_diverging else None,     # PPS does not need centering
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5, "label": cbar_label}
    )
    plt.title(title, pad=20)
    save_plot(filename)
