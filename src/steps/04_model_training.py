# * Final ML -> Results

import logging

from src.shared import config, utils, logger
from src.processing import modeling

logger.setup_logging()
logger = logging.getLogger(__name__)

def main():
    logger.info("=== STEP 04: STARTING MODEL TRAINING ===\n")

    try:
        X, y = utils.load_final_data()
        logger.info(f"Final training data shape: X={X.shape} and y={y.shape}\n")
    except FileNotFoundError as e:
        logger.error(e)
        return

    logger.info("=== STEP 04: COMPLETED SUCCESSFULLY ===\n")

if __name__ == "__main__":
    main()
