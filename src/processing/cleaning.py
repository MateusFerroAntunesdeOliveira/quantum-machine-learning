# * Used for data cleaning operations - like dropping unnecessary columns, handling missing / sentinel values, etc.

import logging

import numpy as np
import pandas as pd

from src.shared import config

# Get logger instance for this module
logger = logging.getLogger(__name__)

def initial_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs initial cleanup:
    1. Standardizes sentinel values (-9999 to NaN).
    2. Renames the ID column (if defined).
    3. Drops columns listed in config.COLS_TO_DROP_UNUSED.
    """
    logger.info("Performing initial cleanup...")

    df = remove_sentinel_values(df, sentinel=[-9999, '-9999'])
    df = rename_raw_id_column(df)
    df = drop_unused_columns(df)
    
    logger.info("Initial cleanup completed.\n")
    return df

def remove_sentinel_values(df: pd.DataFrame, sentinel: list = [-9999, '-9999']) -> pd.DataFrame:
    """
    Replaces specified sentinel values in the DataFrame with NaN.
    """
    logger.info(f"Replacing sentinel values {sentinel} with NaN...")
    df = df.replace({val: np.nan for val in sentinel})
    return df

def rename_raw_id_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the raw ID column to the standardized ID column name defined in config.
    """
    if hasattr(config, 'RAW_ID_COLUMN') and hasattr(config, 'ID_COLUMN'):
        if config.RAW_ID_COLUMN in df.columns:
            logger.info(f"Renaming '{config.RAW_ID_COLUMN}' to '{config.ID_COLUMN}'")
            df = df.rename(columns={config.RAW_ID_COLUMN: config.ID_COLUMN})
    return df

def drop_unused_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that are listed in config.COLS_TO_DROP_UNUSED.
    """
    cols_to_drop = [col for col in config.COLS_TO_DROP_INITIALLY if col in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping {len(cols_to_drop)} unused columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    return df

def drop_columns_by_threshold(df: pd.DataFrame, report: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """
    Analyzes missing data and drops columns that exceed the thresholds 
    defined in config.MISSING_THRESHOLDS for each attribute category.

    Returns:
        tuple: (cleaned_df, dropped_columns_list)
    """
    logger.info("Analyzing missing values against thresholds...")
    columns_to_remove = []

    # Iterate over all columns in the dataframe
    for col in df.columns:
        # * Skip ID and Target from dropping logic
        if col == config.ID_COLUMN or col == config.TARGET_COLUMN:
            continue

        # Determine threshold category
        if col in config.CORE_ATTRIBUTES:
            threshold_value = config.MISSING_THRESHOLDS['core']
            threshold_category = 'CORE'
        elif col in config.SUPPORTING_ATTRIBUTES:
            threshold_value = config.MISSING_THRESHOLDS['support']
            threshold_category = 'SUPPORT'
        elif col in config.RARE_ATTRIBUTES:
            threshold_value = config.MISSING_THRESHOLDS['rare']
            threshold_category = 'RARE'
        else:
            threshold_value = config.MISSING_THRESHOLDS['default']
            threshold_category = 'DEFAULT'

        # Check missing percentage
        if col in report.index:
            # Get the percentage value for this column
            pct_missing = report.loc[col, config.PERCENT_MISSING_COLUMN]
            threshold_percent = threshold_value * 100

            # When threshold is exceeded, mark for removal, else keep
            if pct_missing >= threshold_percent:
                logger.debug(f"Drop [{threshold_category}] {col}: {pct_missing:.2f}% missing (Limit: {threshold_percent:.1f}%)")
                columns_to_remove.append(col)
            else:
                logger.debug(f"Keep [{threshold_category}] {col}: {pct_missing:.2f}% missing (Limit: {threshold_percent:.1f}%)")

    # Perform the drop
    if columns_to_remove:
        df_clean = df.drop(columns=columns_to_remove)
        logger.info(f"Successfully analyzed missing values. Dropping {len(columns_to_remove)} columns.")
        return df_clean, columns_to_remove

    logger.info("Successfully analyzed missing values. No columns to drop.")
    return df, []
