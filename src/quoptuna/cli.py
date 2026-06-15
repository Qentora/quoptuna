"""Command-line interface for QuOptuna.

``quoptuna run`` launches the full stack from the installed package: a single
FastAPI/uvicorn process serves the JSON API and the pre-built static frontend
bundled into the wheel, on one port. ``quoptuna run --streamlit`` launches the
legacy Streamlit app.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import socket
import subprocess
import threading
import time
import webbrowser
from pathlib import Path

import typer
import uvicorn
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

BACKEND_BIND_HOST = "0.0.0.0"
DEFAULT_ACCESS_HOST = "localhost"
DEFAULT_PORT = 8000

_POLL_INTERVAL = 0.5
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


def _open_browser_when_ready(url: str, port: int, host: str = DEFAULT_ACCESS_HOST) -> None:
    """Open the browser from a daemon thread once the server accepts connections.

    ``uvicorn.run`` blocks the main thread, so readiness polling and the browser
    launch happen in the background.
    """

    def _worker() -> None:
        if _wait_until_ready(port, host):
            _open_browser(url)

    threading.Thread(target=_worker, daemon=True).start()


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


def print_banner(access_host: str, port: int) -> None:
    """Print a Langflow-style welcome panel with access links."""
    version = _quoptuna_version()
    url = f"http://{access_host}:{port}"
    docs_url = f"{url}/api/docs"

    try:
        star, ok, arrow = ":star2:", "🟢", "→"
        message = (
            f"[bold {_BRAND_COLOR}]Welcome to QuOptuna[/] [bold]v{version}[/bold]\n\n"
            f"{star} GitHub: Star for updates {arrow} {GITHUB_URL}\n"
            f"{star} Docs {arrow} {DOCS_URL}\n\n"
            f"[bold]{ok} App {arrow}[/bold] [link={url}]{url}[/link]\n"
            f"[bold]{ok} API docs {arrow}[/bold] [link={docs_url}]{docs_url}[/link]"
        )
        console.print()
        console.print(Panel.fit(message, border_style=_BRAND_COLOR, padding=(1, 2)))
    except UnicodeEncodeError:
        fallback = (
            f"Welcome to QuOptuna v{version}\n\n"
            f"GitHub: {GITHUB_URL}\n"
            f"Docs:   {DOCS_URL}\n\n"
            f"App      -> {url}\n"
            f"API docs -> {docs_url}"
        )
        console.print()
        console.print(Panel.fit(fallback, border_style=_BRAND_COLOR, padding=(1, 2)))


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
    port: int = DEFAULT_PORT,
    *,
    open_browser: bool = True,
) -> int:
    """Serve the bundled UI and JSON API from one in-package uvicorn process."""
    print_logo()
    port = _resolve_port(port, host)
    url = f"http://{host}:{port}"

    version = _quoptuna_version()
    console.print(f"[bold {_BRAND_COLOR}]Setting up QuOptuna[/] [bold]v{version}[/bold]")
    print_banner(host, port)

    if open_browser:
        _open_browser_when_ready(url, port, host)

    try:
        uvicorn.run(
            "quoptuna.server.main:app",
            host=BACKEND_BIND_HOST,
            port=port,
            log_level="info",
        )
    except KeyboardInterrupt:
        return 0
    return 0


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
    port: int = typer.Option(DEFAULT_PORT, help="Port for the QuOptuna server."),
    no_browser: bool = typer.Option(
        False,
        "--no-browser",
        help="Do not open the browser after the server is ready.",
    ),
) -> None:
    """Run QuOptuna (full stack by default)."""
    if streamlit:
        raise typer.Exit(run_streamlit())
    raise typer.Exit(
        run_fullstack(
            host=host,
            port=port,
            open_browser=not no_browser,
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
