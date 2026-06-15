"""Tests for the ``quoptuna`` Typer CLI.

The runner functions are monkeypatched so no real subprocesses (uv, npm,
streamlit) are spawned; we assert dispatch, option parsing, and the pure
helpers (port resolution, frontend env, banner rendering).
"""

import socket

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
