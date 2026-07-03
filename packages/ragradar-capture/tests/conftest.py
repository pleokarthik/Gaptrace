import logging

import pytest


@pytest.fixture(autouse=True)
def ragradar_home(tmp_path, monkeypatch):
    ragradar_dir = tmp_path / ".ragradar"
    monkeypatch.setattr("ragradar_core.store._ragradar_dir", lambda: ragradar_dir)
    logger = logging.getLogger("ragradar-capture")
    logger.handlers.clear()
    yield tmp_path
    logger.handlers.clear()
