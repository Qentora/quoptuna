"""Tests for the ``quoptuna`` Typer CLI.

The runner functions are monkeypatched so no real server (uvicorn, streamlit)
is launched; we assert dispatch, option parsing, and the pure helpers (port
resolution, banner rendering, browser launching).
"""

import socket
import types

import pytest
from typer.testing import CliRunner

from quoptuna import cli

runner = CliRunner()


@pytest.fixture
def spy_runners(monkeypatch):
    calls = []

    def _fake_fullstack(
        *,
        host="localhost",
        port=8000,
        open_browser=True,
    ):
        calls.append(
            {
                "kind": "fullstack",
                "host": host,
                "port": port,
                "open_browser": open_browser,
            }
        )
        return 0

    def _fake_streamlit():
        calls.append({"kind": "streamlit"})
        return 0

    monkeypatch.setattr(cli, "run_fullstack", _fake_fullstack)
    monkeypatch.setattr(cli, "run_streamlit", _fake_streamlit)
    return calls


def test_run_defaults_to_fullstack(spy_runners):
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == 0
    assert spy_runners == [
        {
            "kind": "fullstack",
            "host": "localhost",
            "port": 8000,
            "open_browser": True,
        }
    ]


def test_no_args_runs_fullstack(spy_runners):
    result = runner.invoke(cli.app, [])
    assert result.exit_code == 0
    assert spy_runners == [
        {
            "kind": "fullstack",
            "host": "localhost",
            "port": 8000,
            "open_browser": True,
        }
    ]


def test_run_streamlit_flag(spy_runners):
    result = runner.invoke(cli.app, ["run", "--streamlit"])
    assert result.exit_code == 0
    assert spy_runners == [{"kind": "streamlit"}]


def test_run_no_browser_and_custom_port(spy_runners):
    custom_port = 9001
    result = runner.invoke(
        cli.app,
        ["run", "--no-browser", "--port", str(custom_port)],
    )
    assert result.exit_code == 0
    assert spy_runners[0]["open_browser"] is False
    assert spy_runners[0]["port"] == custom_port


def test_run_propagates_exit_code(monkeypatch):
    expected_code = 3
    monkeypatch.setattr(cli, "run_fullstack", lambda **_kwargs: expected_code)
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == expected_code


def test_print_banner_renders(capsys):
    cli.print_banner("localhost", 8000)
    out = capsys.readouterr().out
    assert "QuOptuna" in out


def test_print_logo_renders(capsys):
    cli.print_logo()
    out = capsys.readouterr().out
    assert out.strip()


def test_print_logo_fallback(monkeypatch, capsys):
    def _boom(*_args, **_kwargs):
        raise RuntimeError

    monkeypatch.setattr(cli, "_gradient_color", _boom)
    cli.print_logo()
    out = capsys.readouterr().out
    assert "QuOptuna" in out


def test_gradient_color_endpoints():
    assert cli._gradient_color(0) == cli._GRADIENT_STOPS[0]
    assert cli._gradient_color(1) == cli._GRADIENT_STOPS[-1]


def test_resolve_port_returns_same_when_free():
    # Find a definitely-free port, then confirm _resolve_port keeps it.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("localhost", 0))
        free_port = probe.getsockname()[1]
    assert cli._resolve_port(free_port, "localhost") == free_port


def test_resolve_port_skips_occupied_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("localhost", 0))
        occupied.listen(1)
        occupied_port = occupied.getsockname()[1]
        resolved = cli._resolve_port(occupied_port, "localhost")
        assert resolved != occupied_port
        assert resolved > occupied_port


def test_streamlit_app_path_points_at_app_module():
    path = cli._streamlit_app_path()
    assert path.name == "app.py"
    assert path.parent.name == "frontend"


# ---------------------------------------------------------------------------
# Helper coverage
# ---------------------------------------------------------------------------


def test_quoptuna_version_returns_string():
    assert isinstance(cli._quoptuna_version(), str)


def test_quoptuna_version_unknown(monkeypatch):
    def _raise(_name):
        raise cli.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(cli.importlib.metadata, "version", _raise)
    assert cli._quoptuna_version() == "unknown"


def test_streamlit_app_path_raises_when_missing(monkeypatch):
    monkeypatch.setattr(cli.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError):
        cli._streamlit_app_path()


def test_wait_until_ready_polls_until_open(monkeypatch):
    results = iter([False, True])
    monkeypatch.setattr(cli, "_is_port_in_use", lambda *_a, **_k: next(results, True))
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert cli._wait_until_ready(1234, timeout=10.0) is True


def test_wait_until_ready_times_out(monkeypatch):
    monkeypatch.setattr(cli, "_is_port_in_use", lambda *_a, **_k: False)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert cli._wait_until_ready(1234, timeout=0.0) is False


def test_open_browser_success(monkeypatch):
    opened = []
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    cli._open_browser("http://example.test")
    assert opened == ["http://example.test"]


def test_open_browser_handles_error(monkeypatch, capsys):
    def _raise(_url):
        raise OSError

    monkeypatch.setattr(cli.webbrowser, "open", _raise)
    cli._open_browser("http://example.test")
    assert "http://example.test" in capsys.readouterr().out


def test_open_browser_when_ready_opens_when_up(monkeypatch):
    opened = []

    class _ImmediateThread:
        def __init__(self, *, target, daemon=False):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(cli.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(cli, "_wait_until_ready", lambda *_a, **_k: True)
    monkeypatch.setattr(cli, "_open_browser", opened.append)
    cli._open_browser_when_ready("http://example.test", 8000, "localhost")
    assert opened == ["http://example.test"]


def test_open_browser_when_ready_skips_when_not_up(monkeypatch):
    opened = []

    class _ImmediateThread:
        def __init__(self, *, target, daemon=False):
            self._target = target

        def start(self):
            self._target()

    monkeypatch.setattr(cli.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(cli, "_wait_until_ready", lambda *_a, **_k: False)
    monkeypatch.setattr(cli, "_open_browser", opened.append)
    cli._open_browser_when_ready("http://example.test", 8000, "localhost")
    assert opened == []


def test_print_banner_unicode_fallback(monkeypatch, capsys):
    calls = {"n": 0}
    real_fit = cli.Panel.fit

    def flaky_fit(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            err = UnicodeEncodeError("utf-8", "", 0, 1, "boom")
            raise err
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(cli.Panel, "fit", flaky_fit)
    cli.print_banner("localhost", 8000)
    assert "QuOptuna" in capsys.readouterr().out


def test_main_invokes_app(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(cli, "app", lambda: called.__setitem__("n", 1))
    cli.main()
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# Runner coverage
# ---------------------------------------------------------------------------


def test_run_streamlit_invokes_subprocess(monkeypatch):
    monkeypatch.setattr(cli, "print_logo", lambda: None)
    monkeypatch.setattr(cli, "_streamlit_app_path", lambda: cli.Path("app.py"))
    monkeypatch.setattr(
        cli.subprocess, "run", lambda *_a, **_k: types.SimpleNamespace(returncode=0)
    )
    assert cli.run_streamlit() == 0


def test_run_streamlit_keyboard_interrupt(monkeypatch):
    monkeypatch.setattr(cli, "print_logo", lambda: None)
    monkeypatch.setattr(cli, "_streamlit_app_path", lambda: cli.Path("app.py"))

    def _raise(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.subprocess, "run", _raise)
    assert cli.run_streamlit() == 0


def _setup_fullstack(monkeypatch):
    monkeypatch.setattr(cli, "_resolve_port", lambda port, host=cli.DEFAULT_ACCESS_HOST: port)
    monkeypatch.setattr(cli, "print_logo", lambda: None)
    monkeypatch.setattr(cli, "print_banner", lambda *_a, **_k: None)


def test_run_fullstack_runs_uvicorn(monkeypatch):
    expected_port = 8000
    _setup_fullstack(monkeypatch)
    calls = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: calls.update(app=app, **kwargs))
    monkeypatch.setattr(cli, "_open_browser_when_ready", lambda *_a, **_k: None)
    assert cli.run_fullstack(port=expected_port, open_browser=False) == 0
    assert calls["app"] == "quoptuna.server.main:app"
    assert calls["port"] == expected_port
    assert calls["host"] == cli.BACKEND_BIND_HOST


def test_run_fullstack_opens_browser(monkeypatch):
    _setup_fullstack(monkeypatch)
    opened = {"n": 0}
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli, "_open_browser_when_ready", lambda *_a, **_k: opened.__setitem__("n", 1)
    )
    assert cli.run_fullstack(open_browser=True) == 0
    assert opened["n"] == 1


def test_run_fullstack_no_browser_skips_launch(monkeypatch):
    _setup_fullstack(monkeypatch)
    opened = {"n": 0}
    monkeypatch.setattr(cli.uvicorn, "run", lambda *_a, **_k: None)
    monkeypatch.setattr(
        cli, "_open_browser_when_ready", lambda *_a, **_k: opened.__setitem__("n", 1)
    )
    assert cli.run_fullstack(open_browser=False) == 0
    assert opened["n"] == 0


def test_run_fullstack_keyboard_interrupt(monkeypatch):
    _setup_fullstack(monkeypatch)
    monkeypatch.setattr(cli, "_open_browser_when_ready", lambda *_a, **_k: None)

    def _raise(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.uvicorn, "run", _raise)
    assert cli.run_fullstack() == 0
