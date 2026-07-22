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


@app.command()
def optimize(  # noqa: PLR0913
    uci_id: str = typer.Option(
        None, "--uci-id", help="UCI dataset id (e.g. 53 for Iris). Mutually exclusive with --csv."
    ),
    csv: str = typer.Option(None, "--csv", help="Path to a local CSV dataset."),
    target: str = typer.Option(
        None, "--target", help="Target column (default: the dataset's target / last column)."
    ),
    features: str = typer.Option(
        None, "--features", help="Comma-separated feature columns (default: all non-target)."
    ),
    trials: int = typer.Option(3, "--trials", min=1, help="Number of Optuna trials."),
    models: str = typer.Option(
        "SVC",
        "--models",
        help="Comma-separated model types, e.g. 'SVC,IQPKernelClassifier'.",
    ),
    label_neg: str = typer.Option(None, "--label-neg", help="Binary targets: label mapped to -1."),
    label_pos: str = typer.Option(None, "--label-pos", help="Binary targets: label mapped to +1."),
    favorable_class: str = typer.Option(
        None,
        "--favorable-class",
        help="Multiclass: favorable outcome for fairness auditing (only needed with fairness).",
    ),
    sensitive_feature: str = typer.Option(
        None, "--sensitive-feature", help="Protected attribute column for fairness auditing."
    ),
    fairness_mode: str = typer.Option(
        "off", "--fairness-mode", help="off | constrained | multi_objective."
    ),
    fairness_metric: str = typer.Option(
        "equal_opportunity_difference", "--fairness-metric", help="Disparity metric."
    ),
    fairness_threshold: float = typer.Option(
        None, "--fairness-threshold", help="Constrained-mode disparity threshold."
    ),
    sampler: str = typer.Option("random", "--sampler", help="tpe | random | grid."),
    sampler_seed: int = typer.Option(0, "--seed", help="Sampler seed (reproducible runs)."),
    pruner: str = typer.Option("none", "--pruner", help="none | asha | hyperband."),
    max_steps: int = typer.Option(
        20, "--max-steps", help="Training-step cap for iterative quantum models."
    ),
    convergence_interval: int = typer.Option(
        5, "--convergence-interval", help="Flat-loss convergence window."
    ),
    max_vmap: int = typer.Option(None, "--max-vmap", help="Circuit vectorization width."),
    categorical_encoding: str = typer.Option(
        "ordinal", "--categorical-encoding", help="ordinal | onehot."
    ),
    study_name: str = typer.Option(None, "--study-name", help="Optuna study name."),
    db_name: str = typer.Option("cli_runs", "--db-name", help="Optuna storage database name."),
    subset_size: int = typer.Option(30, "--subset-size", help="Analysis subset size."),
    no_analyze: bool = typer.Option(
        False, "--no-analyze", help="Skip the post-run analysis summary."
    ),
) -> None:
    """Run one optimization through the exact UI pipeline, headless.

    Examples:
        quoptuna optimize --uci-id 53 --trials 2 --models SVC,IQPKernelClassifier
        quoptuna optimize --csv data.csv --target label --trials 3
    """
    import json  # noqa: PLC0415

    if bool(uci_id) == bool(csv):
        console.print("[red]Provide exactly one of --uci-id or --csv.[/red]")
        raise typer.Exit(2)

    # Imported lazily: pulls in the whole server/JAX stack.
    from quoptuna.server.services.headless import run_headless_optimization  # noqa: PLC0415

    print_logo()
    console.print(f"[bold {_BRAND_COLOR}]Running optimization[/] ({trials} trials)")
    try:
        summary = run_headless_optimization(
            csv_path=csv,
            uci_id=uci_id,
            target=target,
            features=[f.strip() for f in features.split(",")] if features else None,
            n_trials=trials,
            model_types=[m.strip() for m in models.split(",")] if models else None,
            label_neg=label_neg,
            label_pos=label_pos,
            favorable_class=favorable_class,
            sensitive_feature=sensitive_feature,
            categorical_encoding=categorical_encoding,
            sampler=sampler,
            sampler_seed=sampler_seed,
            pruner=pruner,
            fairness_mode=fairness_mode,
            fairness_metric=fairness_metric,
            fairness_threshold=fairness_threshold,
            max_steps=max_steps,
            convergence_interval=convergence_interval,
            max_vmap=max_vmap,
            study_name=study_name,
            db_name=db_name,
            analyze=not no_analyze,
            subset_size=subset_size,
        )
    except Exception as exc:
        console.print(f"[red]Optimization failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    console.print_json(json.dumps(summary, default=str))


@app.command("migrate-supabase")
def migrate_supabase(
    source_db: str = typer.Option("db/quoptuna_app.db", "--source-db"),
    database_url: str = typer.Option(None, "--database-url"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Migrate legacy application metadata into SQLModel/PostgreSQL."""
    import json  # noqa: PLC0415

    from quoptuna.server.core.config import settings  # noqa: PLC0415
    from quoptuna.server.services.migration import migrate_app_store  # noqa: PLC0415

    target = database_url or settings.DATABASE_URL
    result = migrate_app_store(source_db, target, dry_run=dry_run)
    console.print_json(json.dumps(result))


def main() -> None:
    from quoptuna.backend.utils.log_file import attach_file_logging  # noqa: PLC0415

    attach_file_logging()
    app()


if __name__ == "__main__":
    main()
