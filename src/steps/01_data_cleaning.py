# * Raw -> Imputed CSV

import logging

from src.shared import config, utils, logger
from src.processing import cleaning, imputation

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 01: STARTING DATA CLEANING AND IMPUTATION ===\n")

    raw_df = utils.load_raw_data()
    df = cleaning.initial_cleanup(raw_df)

    # * Initial missing values report
    initial_missing_values_report = utils.get_missing_value_report(df)
    logger.info(f"Initial missing value report shape: {initial_missing_values_report.shape}")

    # * Drop columns by threshold
    df_clean, dropped = cleaning.drop_columns_by_threshold(df=df, report=initial_missing_values_report)
    logger.info(f"Columns dropped: {len(dropped)}")
    logger.info(f"Columns remaining after drop: {df_clean.shape[1]}")

    # * Remaining missing values report
    remaining_missing_values_report = utils.get_missing_value_report(df_clean)
    logger.info(f"Remaining missing value report shape: {remaining_missing_values_report.shape}")
    # plot_missing_distribution(remaining_missing_values_report)
    utils.save_data(remaining_missing_values_report, config.DROPPED_COLS_FILE, index_value=True, label="remaining missing values report")

    # * Impute missing values
    df_imputed = imputation.run_advanced_imputation(df_clean)

    # * Ensure TARGET_COLUMN (DX_GROUP) is integer after imputation    
    if config.TARGET_COLUMN in df_imputed.columns:
        df_imputed[config.TARGET_COLUMN] = df_imputed[config.TARGET_COLUMN].astype(int)  

    utils.save_data(df_imputed, config.IMPUTED_DATA_FILE, index_value=False, label="Imputed Dataset")

    logger.info("=== STEP 01: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
