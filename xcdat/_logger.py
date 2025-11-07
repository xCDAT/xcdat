"""Logger module for setting up a logger."""

import logging
import logging.handlers

# Logging module setup
LOG_FORMAT = (
    "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
)
LOG_LEVEL = logging.INFO


def _setup_root_logger():
    """Configures the root logger.

    This function sets up the root logger with a predefined format and log level.
    It also enables capturing of warnings issued by the `warnings` module and
    redirects them to the logging system.

    Notes
    -----
    - The `force=True` parameter ensures that any existing logging configuration
      is overridden.
    """
    logging.basicConfig(
        format=LOG_FORMAT,
        level=LOG_LEVEL,
        force=True,
    )
    logging.captureWarnings(True)


def _setup_custom_logger(name, propagate=True) -> logging.Logger:
    """Sets up a custom logger.

    Documentation on logging: https://docs.python.org/3/library/logging.html

    Parameters
    ----------
    name : str
        Name of the file where this function is called.
    propagate : bool, optional
        Whether to propagate logger messages or not, by default False

    Returns
    -------
    logging.Logger
        The logger.

    Examples
    ---------
    Detailed information, typically of interest only when diagnosing problems:

    >>> logger.debug("")

    Confirmation that things are working as expected:

    >>> logger.info("")

    An indication that something unexpected happened, or indicative of some
    problem in the near future:

    >>> logger.warning("")

    The software has not been able to perform some function due to a more
    serious problem:

    >>> logger.error("")

    A serious error, indicating that the program itself may be unable to
    continue running:

    >>> logger.critical("")
    """
    logger = logging.getLogger(name)
    logger.propagate = propagate

    return logger
