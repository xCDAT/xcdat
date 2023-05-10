"""Logger module for setting up a logger."""
import logging
import logging.handlers

# Logging module setup
log_format = (
    "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
)
logging.basicConfig(format=log_format, filemode="w", level=logging.INFO)

# Console handler setup
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logFormatter = logging.Formatter(log_format)
console_handler.setFormatter(logFormatter)
logging.getLogger().addHandler(console_handler)


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
