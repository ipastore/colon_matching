import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter
from datetime import datetime
import hashlib
import os


def setup_logging():
    # Set up logging
    logger = getLogger('debug_logger')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent messages from propagating to the root logger

    # Create console handler for debug logging
    console_handler = StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler for debug logging
    debug_file_handler = FileHandler('logs/debug.log')
    debug_file_handler.setLevel(logging.DEBUG)
    debug_file_handler.setFormatter(console_formatter)
    logger.addHandler(debug_file_handler)

    # Create file handler for info logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'logs/no_matches_{timestamp}.log'
    info_file_handler = FileHandler(log_filename)
    info_file_handler.setLevel(logging.INFO)
    info_formatter = Formatter('%(asctime)s - %(message)s')
    info_file_handler.setFormatter(info_formatter)

    # Add a filter to only log 'No matches found' messages to the info log
    class NoMatchesFilter(logging.Filter):
        def filter(self, record):
            return 'No matches found' in record.getMessage()

    info_file_handler.addFilter(NoMatchesFilter())
    logger.addHandler(info_file_handler)

    return logger

# Ensure the logs directory exists
os.makedirs('logs', exist_ok=True)
