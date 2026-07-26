from pathlib import Path

import pytest

from quoptuna.infra_tui.runner import redact_output, script_command, validate_env_file


def test_script_command_is_allowlisted(tmp_path: Path) -> None:
    assert script_command("status", "dev", terraform_dir=tmp_path) == [
        str(tmp_path / "scripts/status.sh"), "dev"
    ]


def test_script_command_rejects_invalid_action(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        script_command("terraform", "dev", terraform_dir=tmp_path)


def test_redact_output_hides_database_url() -> None:
    assert redact_output("DATABASE_URL=postgres://secret") == "DATABASE_URL=***"


def test_validate_env_file(tmp_path: Path) -> None:
    assert validate_env_file(None)[0]
    assert not validate_env_file(tmp_path / "missing.env")[0]
