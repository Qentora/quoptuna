import logging

import pytest

from quoptuna.backend.utils import log_file


@pytest.fixture(autouse=True)
def _fresh_handler_state(monkeypatch, tmp_path):
    monkeypatch.setattr(log_file, "_attached", None)
    monkeypatch.setenv("QUOPTUNA_LOG_FILE", str(tmp_path / "quoptuna.log"))
    yield
    # Detach anything this test attached so other tests' logging is unaffected.
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "baseFilename", "").startswith(str(tmp_path)):
            root.removeHandler(handler)
            handler.close()


def test_attach_creates_file_and_captures_records(tmp_path):
    path = log_file.attach_file_logging()
    assert path == tmp_path / "quoptuna.log"
    logging.getLogger("quoptuna.test").warning("hello-from-test")
    content = path.read_text(encoding="utf-8")
    assert "hello-from-test" in content
    assert "quoptuna.test" in content


def test_attach_is_idempotent():
    before = len(logging.getLogger().handlers)
    first = log_file.attach_file_logging()
    after_first = len(logging.getLogger().handlers)
    second = log_file.attach_file_logging()
    assert first == second
    assert after_first == before + 1
    assert len(logging.getLogger().handlers) == after_first
