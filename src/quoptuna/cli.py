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
import tempfile
import time
import webbrowser
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

BACKEND_BIND_HOST = "0.0.0.0"
DEFAULT_ACCESS_HOST = "localhost"
DEFAULT_BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 3000

_POLL_INTERVAL = 0.5
_TERMINATE_GRACE = 5.0
_READY_TIMEOUT = 120.0
_BRAND_COLOR = "#9b59b6"

# Gemini-CLI-style horizontal gradient (blue -> purple -> pink).
_GRADIENT_STOPS = ("#4796E4", "#847ACE", "#C3677F")

_LOGO = r"""
 ██████╗ ██╗   ██╗ ██████╗ ██████╗ ████████╗██╗   ██╗███╗   ██╗ █████╗
██╔═══██╗██║   ██║██╔═══██╗██╔══██╗╚══██╔══╝██║   ██║████╗  ██║██╔══██╗
██║   ██║██║   ██║██║   ██║██████╔╝   ██║   ██║   ██║██╔██╗ ██║███████║
██║▄▄ ██║██║   ██║██║   ██║██╔═══╝    ██║   ██║   ██║██║╚██╗██║██╔══██║
╚██████╔╝╚██████╔╝╚██████╔╝██║        ██║   ╚██████╔╝██║ ╚████║██║  ██║
 ╚══▀▀═╝  ╚═════╝  ╚═════╝ ╚═╝        ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝
"""

GITHUB_URL = "https://github.com/Qentora/quoptuna"
DOCS_URL = "https://Qentora.github.io/quoptuna"

console = Console()

app = typer.Typer(
    add_completion=True,
    help="QuOptuna command-line interface.",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quoptuna_version() -> str:
    try:
        return importlib.metadata.version("quoptuna")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _log_dir() -> Path:
    """Directory for backend/frontend log files (created if missing)."""
    path = Path(tempfile.gettempdir()) / "quoptuna"
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def _lerp_hex(start: str, end: str, t: float) -> str:
    """Linear interpolation between two ``#rrggbb`` colors at fraction ``t``."""
    sr, sg, sb = (int(start[i : i + 2], 16) for i in (1, 3, 5))
    er, eg, eb = (int(end[i : i + 2], 16) for i in (1, 3, 5))
    r, g, b = (round(s + (e - s) * t) for s, e in ((sr, er), (sg, eg), (sb, eb)))
    return f"#{r:02x}{g:02x}{b:02x}"


def _gradient_color(t: float) -> str:
    """Color at fraction ``t`` of the multi-stop ``_GRADIENT_STOPS`` ramp."""
    stops = _GRADIENT_STOPS
    if t <= 0:
        return stops[0]
    if t >= 1:
        return stops[-1]
    segment = t * (len(stops) - 1)
    index = int(segment)
    return _lerp_hex(stops[index], stops[index + 1], segment - index)


def print_logo() -> None:
    """Print the QuOptuna block typography with a Gemini-style gradient."""
    try:
        lines = _LOGO.strip("\n").splitlines()
        width = max((len(line) for line in lines), default=1)
        span = max(width - 1, 1)
        for line in lines:
            text = Text()
            for col, char in enumerate(line):
                if char == " ":
                    text.append(" ")
                else:
                    text.append(char, style=_gradient_color(col / span))
            console.print(text)
    except Exception:
        console.print(f"[bold {_BRAND_COLOR}]QuOptuna[/]")


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


def print_log_locations(backend_log: Path, frontend_log: Path) -> None:
    console.print(
        f"[dim]Logs:\n  backend  -> {backend_log}\n  frontend -> {frontend_log}[/dim]",
    )


def _print_log_tail(path: Path, lines: int = 20) -> None:
    """Print the last few lines of a log file for quick debugging."""
    try:
        tail = path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]
    except OSError:
        return
    if tail:
        console.print("[dim]" + "\n".join(tail) + "[/dim]")


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
    print_logo()
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
    print_logo()
    cwd = Path.cwd()
    backend_dir = cwd / "backend"
    frontend_dir = cwd / "frontend"
    _require_dir(backend_dir, "backend")
    _require_dir(frontend_dir, "frontend")

    backend_port = _resolve_port(backend_port, host)
    frontend_port = _resolve_port(frontend_port, host)
    frontend_env = _frontend_env(host, backend_port)

    log_dir = _log_dir()
    build_log = log_dir / "frontend-build.log"
    backend_log = log_dir / "backend.log"
    frontend_log = log_dir / "frontend.log"

    version = _quoptuna_version()
    console.print(f"[bold {_BRAND_COLOR}]Setting up QuOptuna[/] [bold]v{version}[/bold]")
    console.print(f"[green]done[/green] Ports: backend :{backend_port}, frontend :{frontend_port}")

    with console.status("Setting up frontend (building)..."), build_log.open("w") as build_fh:
        build = subprocess.run(
            ["npm", "run", "build"],
            cwd=frontend_dir,
            env=frontend_env,
            stdout=build_fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if build.returncode != 0:
        console.print(f"[red]error:[/red] frontend build failed. See {build_log}")
        _print_log_tail(build_log)
        return build.returncode
    console.print("[green]done[/green] Frontend built")

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
    log_handles = [backend_log.open("w"), frontend_log.open("w")]
    try:
        console.print("[green]done[/green] Starting backend")
        processes.append(
            subprocess.Popen(
                backend_cmd, cwd=backend_dir, stdout=log_handles[0], stderr=subprocess.STDOUT
            )
        )
        console.print("[green]done[/green] Starting frontend")
        processes.append(
            subprocess.Popen(
                frontend_cmd,
                cwd=frontend_dir,
                env=frontend_env,
                stdout=log_handles[1],
                stderr=subprocess.STDOUT,
            )
        )

        with console.status("Waiting for services to be ready..."):
            ready = _wait_until_ready(frontend_port, host)
        if ready:
            print_banner(host, frontend_port, backend_port)
            print_log_locations(backend_log, frontend_log)
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
        for handle in log_handles:
            handle.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def _root(ctx: typer.Context) -> None:
    """QuOptuna command-line interface."""
    if ctx.invoked_subcommand is None:
        raise typer.Exit(run_fullstack())


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
