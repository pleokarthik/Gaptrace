import logging

import pytest


@pytest.fixture(autouse=True)
def gaptrace_home(tmp_path, monkeypatch):
    gaptrace_dir = tmp_path / ".gaptrace"
    monkeypatch.setattr("gaptrace_core.store._gaptrace_dir", lambda: gaptrace_dir)
    logger = logging.getLogger("gaptrace-capture")
    logger.handlers.clear()
    yield tmp_path
    logger.handlers.clear()
