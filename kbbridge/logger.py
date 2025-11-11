import logging
import os


def setup_logging():
    """Configure logging based on environment variables."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logging.basicConfig(
        level=log_level_map.get(log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            *(
                [
                    logging.FileHandler(
                        os.getenv("LOG_FILE", "kbbridge_server.log"), mode="a"
                    )
                ]
                if os.getenv("LOG_TO_FILE", "false").lower() == "true"
                else []
            ),
        ],
    )
    return logging.getLogger(__name__)
