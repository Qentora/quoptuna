"""Textual UI for Terraform-backed QuOptuna infrastructure operations."""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, Binding, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, Header, Label, RichLog, Static
from textual.worker import Worker

from .runner import ALLOWED_ACTIONS, OperationResult, run_operation, validate_env_file

if TYPE_CHECKING:
    from pathlib import Path


class InfraApp(App[None]):
    TITLE = "QuOptuna Infrastructure"
    CSS = """
    Screen { layout: vertical; }
    #body { height: 1fr; }
    #actions { width: 24; border: round $accent; }
    #main { width: 1fr; }
    #status { height: 8; border: round $accent; padding: 1; }
    #logs { height: 1fr; border: round $accent; }
    Button { margin: 1; }
    """
    BINDINGS: ClassVar[Sequence[Binding | tuple[str, str] | tuple[str, str, str]]] = [
        ("q", "quit", "Quit"), ("r", "refresh", "Refresh status")
    ]

    def __init__(self, environment: str, terraform_dir: Path, env_file: Path | None) -> None:
        super().__init__()
        self.environment = environment
        self.terraform_dir = terraform_dir
        self.env_file = env_file
        self.running = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="body"):
            with Vertical(id="actions"):
                yield Label(f"Environment: {self.environment}")
                for action in ALLOWED_ACTIONS:
                    yield Button(action.title(), id=f"action-{action}")
            with Vertical(id="main"):
                yield Static("Status: not checked", id="status")
                yield RichLog(highlight=True, markup=True, id="logs")
        yield Footer()

    def on_mount(self) -> None:
        self.action_refresh()

    def action_refresh(self) -> None:
        self._start_operation("status")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        action = event.button.id.removeprefix("action-") if event.button.id else ""
        if action in ALLOWED_ACTIONS:
            self._start_operation(action)

    def _start_operation(self, action: str) -> None:
        if self.running:
            self.query_one("#logs", RichLog).write("[yellow]Another operation is running.[/yellow]")
            return
        valid, message = validate_env_file(self.env_file)
        log = self.query_one("#logs", RichLog)
        log.write(message)
        if not valid:
            return
        self.running = True
        self.query_one("#status", Static).update(f"Running {action} for {self.environment}...")
        self.run_worker(lambda: self._run(action), exclusive=True, thread=True)

    def _run(self, action: str) -> OperationResult:
        return run_operation(
            action, self.environment, terraform_dir=self.terraform_dir, env_file=self.env_file,
            on_output=self._write_log_from_thread,
        )

    def _write_log_from_thread(self, line: str) -> None:
        self.call_from_thread(self._write_log, line)

    def _write_log(self, line: str) -> None:
        self.query_one("#logs", RichLog).write(line)

    def on_worker_state_changed(self, event: Worker.StateChanged) -> None:
        if event.state.is_terminal:
            self.running = False
            result = event.worker.result
            if isinstance(result, OperationResult):
                status = "succeeded" if result.succeeded else f"failed ({result.returncode})"
                self.query_one("#status", Static).update(f"{result.action}: {status}")
