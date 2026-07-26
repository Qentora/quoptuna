"""Textual infrastructure console smoke tests."""

import asyncio
from pathlib import Path

from textual.widgets import Button, Static

from quoptuna.infra_tui.app import InfraApp


class IdleInfraApp(InfraApp):
    """Avoid launching real scripts when mounting the UI under test."""

    def on_mount(self) -> None:
        pass


def test_infra_app_mounts_all_actions(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = IdleInfraApp("dev", tmp_path, None)
        async with app.run_test() as pilot:
            await pilot.pause()
            button_ids = {button.id for button in app.query(Button)}
            assert "action-create" in button_ids
            assert "action-pause" in button_ids
            assert "action-destroy" in button_ids

    asyncio.run(exercise())


def test_status_json_is_rendered(tmp_path: Path) -> None:
    async def exercise() -> None:
        app = IdleInfraApp("dev", tmp_path, None)
        async with app.run_test() as pilot:
            app._render_status(
                '{"state":"stopped","health":"unavailable","url":"https://example.com",'
                '"instance_id":"i-123","active_work":0,"image":"image:tag"}'
            )
            await pilot.pause()
            rendered = str(app.query_one("#status", Static).render())
            assert "stopped" in rendered
            assert "https://example.com" in rendered

    asyncio.run(exercise())
