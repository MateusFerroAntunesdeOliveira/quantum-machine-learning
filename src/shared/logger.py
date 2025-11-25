import logging
import sys

from . import config

def setup_logging():
    """
    Configures the root logger with the project-standard format.
    Logs are output to both the console (StreamHandler) and a file (FileHandler).
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE, encoding='utf-8')
        ]
    )
