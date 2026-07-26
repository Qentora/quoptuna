"""Safe subprocess execution for infrastructure scripts."""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

ALLOWED_ACTIONS = ("create", "deploy", "update", "pause", "resume", "status", "destroy")
SECRET_PATTERN = re.compile(
    r"(?i)(password|secret|token|database_url|access_key|private_key)=([^\s]+)"
)


@dataclass(frozen=True)
class OperationResult:
    action: str
    returncode: int
    output: str

    @property
    def succeeded(self) -> bool:
        return self.returncode == 0


def redact_output(value: str) -> str:
    return SECRET_PATTERN.sub(r"\1=***", value)


def script_command(  # noqa: PLR0913
                   action: str, environment: str, *, terraform_dir: Path,
                   env_file: Path | None = None, json_output: bool = False,
                   confirmed_destroy: bool = False) -> list[str]:
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Unsupported infrastructure action: {action}")  # noqa: EM102, TRY003
    if not environment or "/" in environment or ".." in environment:
        raise ValueError("Invalid environment name")  # noqa: EM101, TRY003
    command = [str(terraform_dir / "scripts" / f"{action}.sh"), environment]
    if env_file is not None:
        command.extend(["--env-file", str(env_file)])
    if json_output:
        command.append("--json")
    if confirmed_destroy and action == "destroy":
        command.append("--confirm-destroy")
    return command


def run_operation(action: str, environment: str, *, terraform_dir: Path,
                  env_file: Path | None = None,
                  on_output: Callable[[str], None] | None = None) -> OperationResult:
    command = script_command(
        action,
        environment,
        terraform_dir=terraform_dir,
        env_file=env_file,
        json_output=action == "status",
        confirmed_destroy=action == "destroy",
    )
    script = Path(command[0])
    if not script.is_file() or not os.access(script, os.X_OK):
        return OperationResult(action, 127, f"Script is missing or not executable: {script}")
    process = subprocess.Popen(  # noqa: S603
        command, cwd=terraform_dir.parent, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, env=os.environ.copy()
    )
    lines: list[str] = []
    if process.stdout is None:
        return OperationResult(action, 1, "Unable to capture infrastructure script output")
    for line in process.stdout:
        safe_line = redact_output(line.rstrip())
        lines.append(safe_line)
        if on_output is not None:
            on_output(safe_line)
    return OperationResult(action, process.wait(), "\n".join(lines))


def validate_env_file(path: Path | None) -> tuple[bool, str]:
    if path is None:
        return True, "Using process environment"
    if not path.is_file():
        return False, f"Environment file not found: {path}"
    return True, f"Environment file ready: {path}"
