"""Logger module for setting up a logger."""

import logging
import logging.handlers

LOG_FORMAT = (
    "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
)
LOG_LEVEL = logging.INFO


def _setup_xcdat_logger(level: int = LOG_LEVEL, force: bool = False) -> logging.Logger:
    """Configures and returns the xCDAT package logger.

    Parameters
    ----------
    level : int
        Logging level for xCDAT (e.g., logging.DEBUG, logging.INFO).
    force : bool, optional
        If True, clears existing handlers before adding a new one.

    Returns
    -------
    logging.Logger
        The xCDAT package logger.
    """
    logger = logging.getLogger("xcdat")

    if force:
        # Remove all existing handlers (e.g., during reconfiguration)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)

    # Don’t double-log through the root logger
    logger.propagate = False

    return logger


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
