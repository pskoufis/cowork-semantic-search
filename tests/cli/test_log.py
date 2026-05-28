"""Tests for cli.log — logger level mapping and handler idempotency."""

from __future__ import annotations

import logging

import pytest

from cli.log import LOGGER_NAME, configure


@pytest.fixture(autouse=True)
def _reset_logger():
    """Each test starts with a clean library logger so handler state
    accumulated in one test doesn't leak into the next."""
    logger = logging.getLogger(LOGGER_NAME)
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    saved_propagate = logger.propagate
    logger.handlers.clear()
    yield
    logger.handlers.clear()
    for h in saved_handlers:
        logger.addHandler(h)
    logger.setLevel(saved_level)
    logger.propagate = saved_propagate


def test_default_level_is_info():
    logger = configure()
    assert logger.level == logging.INFO


def test_verbose_sets_debug():
    logger = configure(verbose=True)
    assert logger.level == logging.DEBUG


def test_quiet_sets_warning():
    logger = configure(quiet=True)
    assert logger.level == logging.WARNING


def test_verbose_and_quiet_together_raises():
    with pytest.raises(ValueError):
        configure(verbose=True, quiet=True)


def test_configure_is_idempotent():
    """Calling configure() multiple times must not stack handlers — otherwise
    each line would print N times after N invocations."""
    configure()
    configure()
    configure(verbose=True)
    logger = logging.getLogger(LOGGER_NAME)
    owned = [h for h in logger.handlers if getattr(h, "_csemsearch_owned", False)]
    assert len(owned) == 1


def test_propagate_disabled():
    """The library logger must not propagate up to root, otherwise any parent
    handler (e.g. a test runner's root handler) would print every line twice."""
    logger = configure()
    assert logger.propagate is False
