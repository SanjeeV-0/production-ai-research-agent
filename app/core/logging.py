import logging
import sys

from app.config.settings import get_settings


def configure_logging() -> None:
    settings = get_settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

# So the purpose of the whole function is:

# Configure the application's logging behavior from your settings,
#  with a consistent format and output to the terminal.


# logging → Python's built-in logging system.
# sys → used here to access standard output (stdout).

# 2026-08-24 16:45:12 | INFO | app.main | Application started
