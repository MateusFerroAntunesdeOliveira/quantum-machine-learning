# * Imputed -> Plots/Correlations (Exploratory Data Analysis - Optional)

import logging

from src.shared import config, utils, logger
from src.processing import analysis

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    # * Analyze Missing Data (Using Raw and Imputed Data)
    logger.info("--- Part A: Missing Values Analysis ---")
    try:
        df_raw = utils.load_raw_data()
        report = utils.get_missing_value_report(df_raw)
        analysis.plot_missing_distribution(report, fileName="02_missing_values_distribution_raw.png")

        df_imputed = utils.load_imputed_data()
        reportImputed = utils.get_missing_value_report(df_imputed)
        analysis.plot_missing_distribution(reportImputed, fileName="02_missing_values_distribution_imputed.png")

    except FileNotFoundError:
        logger.warning("Raw or Imputed data not found. Skipping missing value plot.")

    # * Correlation & Relationships (Using Imputed Data)
    logger.info("--- Part B: Correlation Analysis (Imputed Data) ---")
    try:
        df_imputed = utils.load_imputed_data()
        logger.info(f"Loaded Imputed Data. Shape: {df_imputed.shape}\n")

        # * Pearson & Spearman
        analysis.compute_and_plot_correlations(df_imputed)

        # * PPS (Predictive Power Score)
        # Note: Depending on data size, you might want to comment this out during quick tests
        analysis.compute_pps_matrix(df_imputed)

    except FileNotFoundError:
        logger.error("Imputed data not found! Please run 'Step 01' first.\n")
        return

    logger.info("=== STEP 02 COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
