"""Tests for the ``quoptuna`` Typer CLI.

The runner functions are monkeypatched so no real subprocesses (uv, npm,
streamlit) are spawned; we assert dispatch, option parsing, and the pure
helpers (port resolution, frontend env, banner rendering).
"""

import socket
import subprocess
import types

import pytest
from typer.testing import CliRunner

from quoptuna import cli

runner = CliRunner()


class FakeProc:
    """Minimal stand-in for ``subprocess.Popen`` used by the runner tests."""

    def __init__(self, polls=None, *, wait_raises=False):
        self._polls = list(polls) if polls is not None else None
        self._wait_raises = wait_raises
        self.terminated = False
        self.killed = False

    def poll(self):
        if self._polls is None:
            return 0
        return self._polls.pop(0) if self._polls else 0

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        if self._wait_raises:
            raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    def kill(self):
        self.killed = True


@pytest.fixture
def spy_runners(monkeypatch):
    calls = []

    def _fake_fullstack(
        *,
        host="localhost",
        backend_port=8000,
        frontend_port=3000,
        open_browser=True,
    ):
        calls.append(
            {
                "kind": "fullstack",
                "host": host,
                "backend_port": backend_port,
                "frontend_port": frontend_port,
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
            "backend_port": 8000,
            "frontend_port": 3000,
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
            "backend_port": 8000,
            "frontend_port": 3000,
            "open_browser": True,
        }
    ]


def test_run_streamlit_flag(spy_runners):
    result = runner.invoke(cli.app, ["run", "--streamlit"])
    assert result.exit_code == 0
    assert spy_runners == [{"kind": "streamlit"}]


def test_run_no_browser_and_custom_ports(spy_runners):
    custom_backend_port = 9001
    custom_frontend_port = 4002
    result = runner.invoke(
        cli.app,
        [
            "run",
            "--no-browser",
            "--backend-port",
            str(custom_backend_port),
            "--frontend-port",
            str(custom_frontend_port),
        ],
    )
    assert result.exit_code == 0
    assert spy_runners[0]["open_browser"] is False
    assert spy_runners[0]["backend_port"] == custom_backend_port
    assert spy_runners[0]["frontend_port"] == custom_frontend_port


def test_run_propagates_exit_code(monkeypatch):
    expected_code = 3
    monkeypatch.setattr(cli, "run_fullstack", lambda **_kwargs: expected_code)
    result = runner.invoke(cli.app, ["run"])
    assert result.exit_code == expected_code


def test_print_banner_renders(capsys):
    cli.print_banner("localhost", 3000, 8000)
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


def test_frontend_env_injects_backend_url():
    backend_port = 8123
    env = cli._frontend_env("localhost", backend_port)
    assert env["NEXT_PUBLIC_API_URL"] == f"http://localhost:{backend_port}"


def test_streamlit_app_path_points_at_app_module():
    path = cli._streamlit_app_path()
    assert path.name == "app.py"
    assert path.parent.name == "frontend"


def test_log_dir_exists():
    path = cli._log_dir()
    assert path.is_dir()


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


def test_require_dir_passes_for_existing(tmp_path):
    cli._require_dir(tmp_path, "backend")


def test_require_dir_raises_for_missing(tmp_path):
    with pytest.raises(SystemExit):
        cli._require_dir(tmp_path / "nope", "backend")


def test_shutdown_kills_on_timeout():
    proc = FakeProc(polls=[None], wait_raises=True)
    cli._shutdown([proc])
    assert proc.terminated is True
    assert proc.killed is True


def test_shutdown_skips_finished_process():
    proc = FakeProc(polls=[0])
    cli._shutdown([proc])
    assert proc.terminated is False


def test_print_log_locations(capsys, tmp_path):
    cli.print_log_locations(tmp_path / "backend.log", tmp_path / "frontend.log")
    out = capsys.readouterr().out
    assert "backend" in out
    assert "frontend" in out


def test_print_log_tail_prints_content(tmp_path, capsys):
    log = tmp_path / "log.txt"
    log.write_text("line1\nline2\n")
    cli._print_log_tail(log)
    assert "line2" in capsys.readouterr().out


def test_print_log_tail_missing_file(tmp_path, capsys):
    cli._print_log_tail(tmp_path / "missing.log")
    assert capsys.readouterr().out == ""


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
    cli.print_banner("localhost", 3000, 8000)
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


def _setup_fullstack(monkeypatch, tmp_path, *, build_returncode, procs):
    (tmp_path / "backend").mkdir(exist_ok=True)
    (tmp_path / "frontend").mkdir(exist_ok=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "_log_dir", lambda: tmp_path)
    monkeypatch.setattr(cli, "_resolve_port", lambda port, host=cli.DEFAULT_ACCESS_HOST: port)
    monkeypatch.setattr(cli, "print_logo", lambda: None)
    monkeypatch.setattr(cli, "print_banner", lambda *_a, **_k: None)
    monkeypatch.setattr(cli, "_open_browser", lambda _url: None)
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *_a, **_k: types.SimpleNamespace(returncode=build_returncode),
    )
    monkeypatch.setattr(cli.subprocess, "Popen", lambda *_a, **_k: procs.pop(0))


def test_run_fullstack_success(monkeypatch, tmp_path):
    procs = [FakeProc(), FakeProc()]
    _setup_fullstack(monkeypatch, tmp_path, build_returncode=0, procs=procs)
    monkeypatch.setattr(cli, "_wait_until_ready", lambda *_a, **_k: True)
    assert cli.run_fullstack(open_browser=True) == 0


def test_run_fullstack_not_ready_skips_browser(monkeypatch, tmp_path):
    procs = [FakeProc(), FakeProc()]
    _setup_fullstack(monkeypatch, tmp_path, build_returncode=0, procs=procs)
    monkeypatch.setattr(cli, "_wait_until_ready", lambda *_a, **_k: False)
    assert cli.run_fullstack(open_browser=False) == 0


def test_run_fullstack_waits_for_process_exit(monkeypatch, tmp_path):
    procs = [FakeProc(polls=[None, 0]), FakeProc(polls=[None])]
    _setup_fullstack(monkeypatch, tmp_path, build_returncode=0, procs=procs)
    monkeypatch.setattr(cli, "_wait_until_ready", lambda *_a, **_k: True)
    monkeypatch.setattr(cli.time, "sleep", lambda _s: None)
    assert cli.run_fullstack(open_browser=False) == 0


def test_run_fullstack_build_failure(monkeypatch, tmp_path):
    _setup_fullstack(monkeypatch, tmp_path, build_returncode=1, procs=[])
    assert cli.run_fullstack() == 1


def test_run_fullstack_keyboard_interrupt(monkeypatch, tmp_path):
    procs = [FakeProc(), FakeProc()]
    _setup_fullstack(monkeypatch, tmp_path, build_returncode=0, procs=procs)

    def _raise(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "_wait_until_ready", _raise)
    assert cli.run_fullstack() == 0


def test_run_fullstack_missing_dir_exits(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "print_logo", lambda: None)
    with pytest.raises(SystemExit):
        cli.run_fullstack()
