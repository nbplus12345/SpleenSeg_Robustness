import logging
import os
import sys


class Logger:
    """Small logger factory used by training scripts."""

    def __init__(self, logger_name="Trainer", log_file="./output/logs/training.log"):
        """Create file and console handlers if the logger is not configured yet."""
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        if not self.logger.handlers:
            log_dir = os.path.dirname(log_file)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

            file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter(
                "[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)

            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(logging.Formatter("%(message)s"))

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        """Return the configured logger instance."""
        return self.logger
