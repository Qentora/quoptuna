"""Regression tests for durable analysis snapshots and snapshot-backed reports."""

import base64
from pathlib import Path

import pytest

from quoptuna.backend.xai import report_agent
from quoptuna.server.api.v1 import analysis
from quoptuna.server.services import analysis_store, run_store


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_store, "ARTIFACT_ROOT", tmp_path / "analysis")
    return tmp_path


def test_snapshot_round_trip_extracts_images_and_revises_same_setup(isolated_store):
    image = b"durable-png"
    data_url = "data:image/png;base64," + base64.b64encode(image).decode()
    config = {"trial_number": 3, "subset_size": 25, "use_proba": True}

    first = analysis_store.create_job("run-1", config)
    completed = analysis_store.complete_job(
        first["id"], {"metrics": {"f1": 0.9}, "plots": {"bar": data_url}}
    )
    snapshot = analysis_store.get_snapshot(completed["snapshot_id"])

    assert snapshot["revision"] == 1
    assert snapshot["payload"]["plots"]["bar"] == data_url
    artifact = next(Path(snapshot["artifact_dir"]).glob("*.png"))
    assert artifact.read_bytes() == image

    second = analysis_store.create_job("run-1", config)
    assert second["snapshot_id"] == first["snapshot_id"]
    analysis_store.complete_job(second["id"], {"metrics": {"f1": 0.95}, "plots": {}})
    revised = analysis_store.get_snapshot(first["snapshot_id"])
    assert revised["revision"] == 2  # noqa: PLR2004
    assert revised["payload"]["metrics"]["f1"] == 0.95  # noqa: PLR2004


def test_different_configuration_gets_separate_snapshot(isolated_store):
    first = analysis_store.create_job("run-1", {"trial_number": 1, "subset_size": 50})
    second = analysis_store.create_job("run-1", {"trial_number": 2, "subset_size": 50})
    assert first["snapshot_id"] != second["snapshot_id"]


def test_duplicate_active_setup_reuses_job(isolated_store):
    first = analysis_store.create_job("run-1", {"trial_number": 1})
    duplicate = analysis_store.create_job("run-1", {"trial_number": 1})
    assert duplicate["id"] == first["id"]
    assert duplicate["created"] is False


def test_reports_persist_and_run_cleanup_removes_artifacts(isolated_store):
    job = analysis_store.create_job("run-1", {})
    analysis_store.complete_job(job["id"], {"metrics": {"accuracy": 1}, "plots": {}})
    snapshot = analysis_store.get_snapshot(job["snapshot_id"])
    report_id = analysis_store.create_report(snapshot, "openai", "test-model", None)
    analysis_store.complete_report(report_id, "# Persisted")

    reports = analysis_store.list_reports(snapshot["id"])
    assert reports[0]["markdown"] == "# Persisted"
    assert "api_key" not in reports[0]

    analysis_store.delete_for_run("run-1")
    assert analysis_store.get_snapshot(snapshot["id"]) is None
    assert not (analysis_store.ARTIFACT_ROOT / "run-1").exists()


@pytest.mark.asyncio
async def test_report_uses_snapshot_without_rebuilding_xai(isolated_store, monkeypatch):
    job = analysis_store.create_job("run-1", {})
    analysis_store.complete_job(
        job["id"],
        {
            "metrics": {"accuracy": 0.8},
            "feature_importance": [],
            "plots": {},
            "confusion_data": {"matrix": [[4, 1], [1, 4]]},
        },
    )
    snapshot = analysis_store.get_snapshot(job["snapshot_id"])

    def forbidden(*args, **kwargs):
        raise AssertionError

    async def fake_generate_report(**kwargs):
        assert kwargs["report"]["metrics"]["accuracy"] == 0.8  # noqa: PLR2004
        return "# Snapshot report"

    monkeypatch.setattr(analysis, "build_xai", forbidden)
    monkeypatch.setattr(report_agent, "generate_report", fake_generate_report)
    response = await analysis.generate_ai_report(
        analysis.ReportRequest(
            optimization_id="run-1",
            analysis_snapshot_id=snapshot["id"],
            analysis_revision=snapshot["revision"],
            api_key="not-persisted",
            llm_provider="openai",
            model_name="test-model",
        )
    )

    assert response["report_markdown"] == "# Snapshot report"
    assert analysis_store.list_reports(snapshot["id"])[0]["markdown"] == "# Snapshot report"
