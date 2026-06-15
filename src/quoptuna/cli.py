"""Command-line interface for QuOptuna.

``quoptuna run`` launches the full stack in production mode (builds the Next.js
frontend then serves it alongside the FastAPI backend), and
``quoptuna run --streamlit`` launches the legacy Streamlit app.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import socket
import subprocess
import time
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

BACKEND_BIND_HOST = "0.0.0.0"
DEFAULT_ACCESS_HOST = "localhost"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000

_POLL_INTERVAL = 0.5
_TERMINATE_GRACE = 5.0
_READY_TIMEOUT = 120.0
_BRAND_COLOR = "#9b59b6"

GITHUB_URL = "https://github.com/Qentora/quoptuna"
DOCS_URL = "https://Qentora.github.io/quoptuna"

console = Console()

app = typer.Typer(
    add_completion=True,
    help="QuOptuna command-line interface.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quoptuna_version() -> str:
    try:
        return importlib.metadata.version("quoptuna")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _streamlit_app_path() -> Path:
    """Resolve the installed location of the Streamlit app module."""
    spec = importlib.util.find_spec("quoptuna.frontend.app")
    if spec is None or spec.origin is None:
        msg = "Could not locate the Streamlit app module 'quoptuna.frontend.app'."
        raise RuntimeError(msg)
    return Path(spec.origin)


def _is_port_in_use(port: int, host: str = DEFAULT_ACCESS_HOST) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def _resolve_port(port: int, host: str = DEFAULT_ACCESS_HOST) -> int:
    """Return the given port if free, otherwise the next free port above it."""
    resolved = port
    while _is_port_in_use(resolved, host):
        resolved += 1
    return resolved


def _frontend_env(access_host: str, backend_port: int) -> dict[str, str]:
    """Frontend process env with the resolved backend URL.

    ``NEXT_PUBLIC_*`` values are inlined at build time, so injecting this into
    both ``npm run build`` and ``npm run start`` keeps the client bundle and the
    SSR runtime pointed at the resolved backend port.
    """
    return {**os.environ, "NEXT_PUBLIC_API_URL": f"http://{access_host}:{backend_port}"}


def _wait_until_ready(
    port: int,
    host: str = DEFAULT_ACCESS_HOST,
    timeout: float = _READY_TIMEOUT,
) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _is_port_in_use(port, host):
            return True
        time.sleep(_POLL_INTERVAL)
    return False


def _open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except (webbrowser.Error, OSError):
        # Opening a browser is best-effort; never fail the run because of it.
        console.print(f"[dim]Could not open a browser automatically. Visit {url}[/dim]")


def print_banner(access_host: str, frontend_port: int, backend_port: int) -> None:
    """Print a Langflow-style welcome panel with access links."""
    version = _quoptuna_version()
    frontend_url = f"http://{access_host}:{frontend_port}"
    backend_url = f"http://{access_host}:{backend_port}"
    docs_url = f"{backend_url}/api/docs"

    try:
        star, ok, arrow = ":star2:", "🟢", "→"
        message = (
            f"[bold {_BRAND_COLOR}]Welcome to QuOptuna[/] [bold]v{version}[/bold]\n\n"
            f"{star} GitHub: Star for updates {arrow} {GITHUB_URL}\n"
            f"{star} Docs {arrow} {DOCS_URL}\n\n"
            f"[bold]{ok} Frontend {arrow}[/bold] [link={frontend_url}]{frontend_url}[/link]\n"
            f"[bold]{ok} API {arrow}[/bold] [link={backend_url}]{backend_url}[/link]\n"
            f"[bold]{ok} API docs {arrow}[/bold] [link={docs_url}]{docs_url}[/link]"
        )
        console.print()
        console.print(Panel.fit(message, border_style=_BRAND_COLOR, padding=(1, 2)))
    except UnicodeEncodeError:
        fallback = (
            f"Welcome to QuOptuna v{version}\n\n"
            f"GitHub: {GITHUB_URL}\n"
            f"Docs:   {DOCS_URL}\n\n"
            f"Frontend -> {frontend_url}\n"
            f"API      -> {backend_url}\n"
            f"API docs -> {docs_url}"
        )
        console.print()
        console.print(Panel.fit(fallback, border_style=_BRAND_COLOR, padding=(1, 2)))


def _require_dir(path: Path, name: str) -> None:
    if not path.is_dir():
        console.print(
            f"[red]error:[/red] '{name}' directory not found at {path}.\n"
            "Run `quoptuna run` from the repository root.",
        )
        raise SystemExit(1)


def _shutdown(processes: list[subprocess.Popen]) -> None:
    """Terminate child processes, escalating to kill after a grace period."""
    for proc in processes:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + _TERMINATE_GRACE
    for proc in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            proc.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_streamlit() -> int:
    """Run the legacy Streamlit interface (blocking, foreground)."""
    app_path = _streamlit_app_path()
    console.print(f"Starting QuOptuna Streamlit app: {app_path}")
    try:
        completed = subprocess.run(["streamlit", "run", str(app_path)], check=False)
    except KeyboardInterrupt:
        return 0
    return completed.returncode


def run_fullstack(
    host: str = DEFAULT_ACCESS_HOST,
    backend_port: int = DEFAULT_BACKEND_PORT,
    frontend_port: int = DEFAULT_FRONTEND_PORT,
    *,
    open_browser: bool = True,
) -> int:
    """Build the frontend, then serve backend + frontend in production mode."""
    cwd = Path.cwd()
    backend_dir = cwd / "backend"
    frontend_dir = cwd / "frontend"
    _require_dir(backend_dir, "backend")
    _require_dir(frontend_dir, "frontend")

    backend_port = _resolve_port(backend_port, host)
    frontend_port = _resolve_port(frontend_port, host)
    frontend_env = _frontend_env(host, backend_port)

    console.print(f"[1/4] Resolved ports (backend :{backend_port}, frontend :{frontend_port})")

    console.print("[2/4] Building Next.js frontend (production)...")
    build = subprocess.run(["npm", "run", "build"], cwd=frontend_dir, env=frontend_env, check=False)
    if build.returncode != 0:
        console.print("[red]error:[/red] frontend build failed.")
        return build.returncode

    backend_cmd = [
        "uv",
        "run",
        "--no-sync",
        "uvicorn",
        "app.main:app",
        "--host",
        BACKEND_BIND_HOST,
        "--port",
        str(backend_port),
    ]
    frontend_cmd = ["npm", "run", "start", "--", "-p", str(frontend_port)]

    processes: list[subprocess.Popen] = []
    try:
        console.print(f"[3/4] Starting backend (:{backend_port})...")
        processes.append(subprocess.Popen(backend_cmd, cwd=backend_dir))
        console.print(f"[4/4] Starting frontend (:{frontend_port})...")
        processes.append(subprocess.Popen(frontend_cmd, cwd=frontend_dir, env=frontend_env))

        if _wait_until_ready(frontend_port, host):
            print_banner(host, frontend_port, backend_port)
            if open_browser:
                _open_browser(f"http://{host}:{frontend_port}")

        while True:
            for proc in processes:
                code = proc.poll()
                if code is not None:
                    return code
            time.sleep(_POLL_INTERVAL)
    except KeyboardInterrupt:
        return 0
    finally:
        _shutdown(processes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.callback()
def _root() -> None:
    """QuOptuna command-line interface."""


@app.command()
def run(
    streamlit: bool = typer.Option(
        False,
        "--streamlit",
        help="Run the legacy Streamlit app instead of the full stack.",
    ),
    host: str = typer.Option(DEFAULT_ACCESS_HOST, help="Access host used in URLs and links."),
    backend_port: int = typer.Option(DEFAULT_BACKEND_PORT, help="Backend (FastAPI) port."),
    frontend_port: int = typer.Option(DEFAULT_FRONTEND_PORT, help="Frontend (Next.js) port."),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not open the browser after the frontend is ready.",
    ),
) -> None:
    """Run QuOptuna (full stack by default)."""
    if streamlit:
        raise typer.Exit(run_streamlit())
    raise typer.Exit(
        run_fullstack(
            host=host,
            backend_port=backend_port,
            frontend_port=frontend_port,
            open_browser=not no_browser,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
