# * Imputed -> Final ML Dataset

import logging

from src.shared import config, utils, logger
# from src.processing import analysis

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 03: STARTING FEATURE ENGINEERING ===\n")

    logger.info("=== STEP 03: COMPLETED SUCCESSFULLY ===\n")


if __name__ == "__main__":
    main()
