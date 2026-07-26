"""Textual UI for Terraform-backed QuOptuna infrastructure operations."""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, ClassVar

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, RichLog, Static
from textual.worker import WorkerState

from .runner import ALLOWED_ACTIONS, OperationResult, run_operation, validate_env_file

if TYPE_CHECKING:
    from pathlib import Path

    from textual.app import Binding
    from textual.worker import Worker


class ConfirmOperation(ModalScreen[bool]):
    """Require confirmation, optionally by typing an exact environment name."""

    CSS = """
    ConfirmOperation {
        align: center middle;
    }
    #confirm-dialog {
        width: 62;
        height: auto;
        border: round $warning;
        background: $surface;
        padding: 1 2;
    }
    #confirm-buttons {
        height: auto;
        align-horizontal: right;
    }
    """

    def __init__(self, action: str, environment: str, *, typed: bool = False) -> None:
        super().__init__()
        self.action_name = action
        self.environment = environment
        self.typed = typed

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label(
                f"{self.action_name.title()} infrastructure for {self.environment}?"
            )
            if self.typed:
                yield Label(f"Type {self.environment} to confirm.")
                yield Input(placeholder=self.environment, id="confirm-input")
            with Horizontal(id="confirm-buttons"):
                yield Button("Cancel", id="cancel")
                yield Button("Confirm", variant="error", id="confirm")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(False)  # noqa: FBT003
            return
        if event.button.id == "confirm":
            if self.typed:
                value = self.query_one("#confirm-input", Input).value.strip()
                if value != self.environment:
                    self.notify("Environment name does not match", severity="error")
                    return
            self.dismiss(True)  # noqa: FBT003


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
    BINDINGS: ClassVar[list[Binding | tuple[str, str] | tuple[str, str, str]]] = [
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
            if action in {"pause", "destroy"}:
                self.push_screen(
                    ConfirmOperation(action, self.environment, typed=action == "destroy"),
                    lambda confirmed: self._confirmed_operation(action, confirmed=confirmed),
                )
            else:
                self._start_operation(action)

    def _confirmed_operation(self, action: str, *, confirmed: bool | None) -> None:
        if confirmed:
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
        if event.state in (WorkerState.CANCELLED, WorkerState.ERROR, WorkerState.SUCCESS):
            self.running = False
            if event.state is not WorkerState.SUCCESS:
                self.query_one("#status", Static).update(f"Operation {event.state.name.lower()}")
                return
            result = event.worker.result
            if isinstance(result, OperationResult):
                status = "succeeded" if result.succeeded else f"failed ({result.returncode})"
                if result.action == "status" and result.succeeded:
                    self._render_status(result.output)
                else:
                    self.query_one("#status", Static).update(f"{result.action}: {status}")

    def _render_status(self, output: str) -> None:
        try:
            value = json.loads(output)
        except json.JSONDecodeError:
            self.query_one("#status", Static).update("Status returned invalid JSON")
            return
        self.query_one("#status", Static).update(
            "\n".join(
                [
                    f"State: {value.get('state', 'unknown')}",
                    f"Health: {value.get('health', 'unknown')}",
                    f"URL: {value.get('url', '-')}",
                    f"Instance: {value.get('instance_id', '-')}",
                    f"Active work: {value.get('active_work', 0)}",
                    f"Image: {value.get('image') or '-'}",
                ]
            )
        )
