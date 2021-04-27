#!/usr/bin/python
# -*- coding: utf-8 -*-

import logging
import logging.handlers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from logging import Logger


def setup_custom_logger(name: str) -> "Logger":
    """Sets up a custom global logger.

    To use, instantiate a logger variable in xcdat/xcdat.py and import in other modules.

    :param name: The name of the logger
    :type name: str
    :return: A Logger object
    :rtype: Logger
    """
    log_format = "%(asctime)s [%(levelname)s]: %(filename)s(%(funcName)s:%(lineno)s) >> %(message)s"
    log_filemode = "w"  # w: overwrite; a: append

    # Setup
    logging.basicConfig(format=log_format, filemode=log_filemode, level=logging.DEBUG)
    logger = logging.getLogger(name)
    logger.propagate = False

    # Console output
    consoleHandler = logging.StreamHandler()
    logFormatter = logging.Formatter(log_format)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

    return logger


# source: https://docs.python.org/2/howto/logging.html
# logger.debug("")      // Detailed information, typically of interest only when diagnosing problems.
# logger.info("")       // Confirmation that things are working as expected.
# logger.warning("")    // An indication that something unexpected happened, or indicative of some problem in the near future
# logger.error("")      // Due to a more serious problem, the software has not been able to perform some function.
# logger.critical("")   // A serious error, indicating that the program itself may be unable to continue running.
