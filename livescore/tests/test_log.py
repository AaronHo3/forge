"""
Unit tests for log.py — the print→logging seam.

These assert the two behaviours the migration relies on:
  - libraries emit through namespaced child loggers of "livescore";
  - with no configure() (the pytest default) records still reach caplog but
    nothing is written to a console handler we added;
  - configure() attaches exactly one console handler, is idempotent, and honours
    the verbose flag.

configure() mutates process-global logging state, so each test that calls it
snapshots and restores the "livescore" logger's handlers/level/propagate and the
module's _configured flag, keeping the suite isolated.
"""

import logging

import pytest

import log


@pytest.fixture
def restore_log_state():
    """Snapshot and restore the global livescore logger + module flag."""
    logger = logging.getLogger("livescore")
    saved = (list(logger.handlers), logger.level, logger.propagate, log._configured)
    try:
        yield logger
    finally:
        logger.handlers = saved[0]
        logger.setLevel(saved[1])
        logger.propagate = saved[2]
        log._configured = saved[3]


@pytest.mark.unit
def test_get_logger_is_namespaced_child():
    assert log.get_logger("mrt").name == "livescore.mrt"
    assert log.get_logger("llm").name == "livescore.llm"


@pytest.mark.unit
def test_package_logger_has_nullhandler():
    """The import-time NullHandler is what makes 'no config => silent'."""
    handlers = logging.getLogger("livescore").handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


@pytest.mark.unit
def test_records_are_capturable_without_configure(caplog):
    """Even with no configure(), child records propagate so caplog (and tests)
    can assert on engine diagnostics."""
    with caplog.at_level(logging.WARNING, logger="livescore"):
        log.get_logger("mrt").warning("[MRT2] captured")
    assert "[MRT2] captured" in caplog.text


@pytest.mark.unit
def test_configure_adds_one_stream_handler_and_is_idempotent(restore_log_state):
    logger = restore_log_state
    before = sum(isinstance(h, logging.StreamHandler)
                 and not isinstance(h, logging.NullHandler)
                 for h in logger.handlers)
    log.configure()
    log.configure()  # second call must not add another handler
    after = sum(isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.NullHandler)
                for h in logger.handlers)
    assert after == before + 1
    assert logger.propagate is False


@pytest.mark.unit
def test_configure_levels(restore_log_state):
    logger = restore_log_state
    log.configure(verbose=False)
    assert logger.level == logging.INFO
    # A repeat call may re-tighten/loosen the level even though it's idempotent
    # about the handler.
    log.configure(verbose=True)
    assert logger.level == logging.DEBUG
