import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter
from datetime import datetime
import os

# Global debug flags available.
DEBUG_FLAGS = {"masking", "error_measurement", "ALL"}

class DebugFlagFilter(logging.Filter):
    """
    A filter that only allows DEBUG messages which include a 'debug_flag'
    attribute that is in the active flags (or if 'ALL' is active).
    """
    def __init__(self, active_flags):
        super().__init__()
        self.active_flags = active_flags

    def filter(self, record):
        # Allow non-debug messages.
        if record.levelno != logging.DEBUG:
            return True

        # For debug-level messages, check for the 'debug_flag' extra attribute.
        flag = getattr(record, 'debug_flag', None)
        if flag is not None and (flag in self.active_flags or "ALL" in self.active_flags):
            return True

        # Filter out the message if the flag is missing or not active.
        return False

def setup_logging(DEBUG, activated_debug_flags):
    """
    Set up the logging configuration:
      - Creates necessary handlers.
      - Attaches a DebugFlagFilter to control which DEBUG messages are logged.
      - Stores the active debug flags on the logger for use in the debug_log wrapper.
    """
    # Ensure the logs directory exists.
    os.makedirs('logs', exist_ok=True)

    # Create and configure the logger.
    logger = getLogger('debug_logger')
    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent propagation to the root logger.

    # Save the active debug flags on the logger for early checking in debug_log.
    logger.active_debug_flags = activated_debug_flags

    # Setup console handler.
    console_handler = StreamHandler()
    if DEBUG:
        console_handler.setLevel(logging.DEBUG)
        console_formatter = Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
    else:
        console_handler.setLevel(logging.INFO)
        console_formatter = Formatter('%(asctime)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for general info logging.
    info_handler = FileHandler('logs/info.log')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(console_formatter)
    logger.addHandler(info_handler)

    # File handler for logging messages that include "No matches found".
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    no_matches_filename = f'logs/no_matches_{timestamp}.log'
    no_matches_handler = FileHandler(no_matches_filename)
    no_matches_handler.setLevel(logging.INFO)
    no_matches_formatter = Formatter('%(asctime)s - %(message)s')
    no_matches_handler.setFormatter(no_matches_formatter)

    # Only log messages that contain "No matches found".
    class NoMatchesFilter(logging.Filter):
        def filter(self, record):
            return 'No matches found' in record.getMessage()

    no_matches_handler.addFilter(NoMatchesFilter())
    logger.addHandler(no_matches_handler)

    # Extra file handler for DEBUG logs (only if DEBUG is True).
    if DEBUG:
        extra_handler = FileHandler('logs/debug.log')
        extra_handler.setLevel(logging.DEBUG)
        extra_formatter = Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(message)s')
        extra_handler.setFormatter(extra_formatter)
        logger.addHandler(extra_handler)
        # Log a startup message via the wrapper to include the debug flag.
        debug_log(logger, 'DEBUG_SETUP', "Extra DEBUG logging is enabled.")

    # Attach our custom filter to the logger.
    logger.addFilter(DebugFlagFilter(activated_debug_flags))
    
    return logger

def debug_log(logger, flag, message, *args, **kwargs):
    """
    Wrapper for logger.debug that:
      - Checks if the logger is None.
      - Checks if the logger is enabled for DEBUG (avoiding unnecessary work).
      - Checks early if the provided debug flag is active.
      - Attaches the 'debug_flag' extra attribute.
    """
    # Check if logger is valid and DEBUG is enabled.
    if logger is None or not logger.isEnabledFor(logging.DEBUG):
        return

    # Early check of the debug flag using the active flags stored on the logger.
    active_flags = getattr(logger, 'active_debug_flags', set())
    if flag not in active_flags and "ALL" not in active_flags:
        return

    # Create an extra dict with just the debug_flag
    extra = {'debug_flag': flag}
    
    # Extract any existing extra values from kwargs
    if 'extra' in kwargs:
        extra.update(kwargs.pop('extra'))
        
    # Log the debug message with the appropriate extra attribute and stacklevel
    logger.debug(message, *args, extra=extra, stacklevel=2, **kwargs)

