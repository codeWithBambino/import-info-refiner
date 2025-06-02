import os
from config.config import LOG_DIR
import logging
from datetime import datetime

def get_timestamp():
    """Returns formatted timestamp with milliseconds."""
    return datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]  # drop last 3 microsecond digits

def setup_logger(folder: str, raw_manifest_filename: str) -> logging.Logger:
    """
    Sets up and returns a logger instance with a timestamped filename.

    Args:
        folder (str): Subfolder under /logs/ for the specific step.
        raw_manifest_filename (str): Name of the CSV file being processed.

    Returns:
        logging.Logger: Configured logger object.
    """
    # Ensure log base and step-specific directories exist
    base_log_dir = os.path.join(LOG_DIR, folder)
    os.makedirs(base_log_dir, exist_ok=True)

    # Create log filename with date only - no sequence number
    base_name = os.path.splitext(raw_manifest_filename)[0]  # remove .csv
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Use a single log file per CSV per day
    log_filename = f"{base_name}_{current_date}.log"
    full_log_path = os.path.join(base_log_dir, log_filename)

    # Set up logger
    logger = logging.getLogger(full_log_path)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Prevent propagation to root logger

    # Avoid adding handlers multiple times if logger is reused
    if not logger.handlers:
        # File handler for all logs
        file_handler = logging.FileHandler(full_log_path)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler for errors only
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

def log_message(folder: str, raw_manifest_filename: str, log_string: str, level: str = "info"):
    """
    Logs a message to the appropriate file.

    Args:
        folder (str): Log subfolder.
        raw_manifest_filename (str): CSV file name being processed.
        log_string (str): Message to log.
        level (str): Logging level ("info", "error", "warning", "debug").
    """
    logger = setup_logger(folder, raw_manifest_filename)

    try:
        # Validate log level
        valid_levels = ["info", "error", "warning", "debug"]
        if level.lower() not in valid_levels:
            raise ValueError(f"Invalid log level: {level}. Must be one of {valid_levels}")

        logger = setup_logger(folder, raw_manifest_filename)
        log_method = getattr(logger, level.lower(), logger.info)
        log_method(log_string)

    except (OSError, IOError) as e:
        # Handle file operation errors
        print(f"ERROR: Failed to write to log file: {str(e)}")
        raise
    except Exception as e:
        print(f"ERROR: Unexpected error in logging: {str(e)}")
        raise