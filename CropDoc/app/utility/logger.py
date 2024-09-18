# Standard Imports
import os
import sys
from datetime import datetime

# Third Party Imports
from loguru import logger

# Local Imports
from app.config import AppConfig


def setup_logger():

    # Obtain logger details from config
    environment = os.getenv("FLASK_ENV", "production")
    diagnose = False if environment == "production" else True
    logging_level = os.getenv("LOGGING_LEVEL", "INFO")

    # Remove the default handler
    logger.remove()

    # Configure Loguru logger
    logger_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>")
    logger.add(sys.stderr,
               colorize=True,
               diagnose=diagnose,
               format=logger_format,
               level=logging_level)

    # Check if directory exists
    if not os.path.exists(f"{AppConfig.CROPDOC_DIR}/data/logs"):
        os.makedirs(f"{AppConfig.CROPDOC_DIR}/data/logs")

    now = datetime.now()
    now_string = now.strftime("%Y-%m-%d_%H-%M-%S")
    # Add a log file
    logger.add(
        f"{AppConfig.CROPDOC_DIR}/data/logs/CropDoc-{now_string}.log",
        diagnose=diagnose,
        retention="7 days",
        enqueue=True,
        #compression="zip",
        format=logger_format,
        level="DEBUG")
