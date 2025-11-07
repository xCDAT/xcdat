import logging

import pytest

from xcdat._logger import _setup_custom_logger, _setup_xcdat_logger


@pytest.fixture(autouse=True)
def reset_logging():
    # Reset logging before each test.
    logging.shutdown()

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Reset xcdat logger specifically
    xcdat_logger = logging.getLogger("xcdat")
    for handler in xcdat_logger.handlers[:]:
        xcdat_logger.removeHandler(handler)

    xcdat_logger.setLevel(logging.NOTSET)
    xcdat_logger.propagate = True

    yield

    # Cleanup after test.
    logging.shutdown()


class TestSetupXCDATLogger:
    def test_logger_creation(self):
        logger = _setup_xcdat_logger()

        assert isinstance(logger, logging.Logger)
        assert logger.name == "xcdat"

    def test_logger_level(self):
        logger = _setup_xcdat_logger(level=logging.DEBUG)

        assert logger.level == logging.DEBUG

    def test_logger_force(self):
        logger = _setup_xcdat_logger()
        handler_ids_before = [id(h) for h in logger.handlers]

        # Force reconfiguration.
        logger = _setup_xcdat_logger(force=True)

        handler_ids_after = [id(h) for h in logger.handlers]

        # Clean reset with one handler.
        assert len(logger.handlers) == 1
        # The handler should be a new instance, not the same object.
        assert handler_ids_before != handler_ids_after

    def test_logger_format(self):
        logger = _setup_xcdat_logger()
        handler = logger.handlers[0]

        assert isinstance(handler.formatter, logging.Formatter)

        # Use the public API to check the formatter's output
        test_record = logging.LogRecord(
            name="xcdat",
            level=logging.INFO,
            pathname=__file__,
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None,
            func="test_func",
        )
        formatted = handler.formatter.format(test_record)
        assert "Test message" in formatted
        assert "[INFO]" in formatted
        assert "test_func" in formatted
        assert str(123) in formatted

    def test_logger_propagation(self):
        logger = _setup_xcdat_logger()

        assert logger.propagate is False

    def test_logger_no_duplicate_handlers(self):
        logger1 = _setup_xcdat_logger()
        num_handlers_before = len(logger1.handlers)

        logger2 = _setup_xcdat_logger()
        num_handlers_after = len(logger2.handlers)

        assert num_handlers_after == num_handlers_before


class TestSetupCustomLogger:
    def test_custom_logger_creation(self):
        logger = _setup_custom_logger("test_logger")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_custom_logger_propagation(self):
        logger = _setup_custom_logger("test_logger", propagate=False)
        assert logger.propagate is False

        logger = _setup_custom_logger("test_logger", propagate=True)
        assert logger.propagate is True

    def test_custom_logger_inherits_from_xcdat(self):
        _setup_xcdat_logger()
        logger = _setup_custom_logger("xcdat.submodule")

        assert logger.parent is not None and logger.parent.name == "xcdat"
