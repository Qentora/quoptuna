"""Regression tests for durable analysis snapshots and snapshot-backed reports."""

import asyncio
import base64
import io
import zipfile
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


def test_report_uses_snapshot_without_rebuilding_xai(isolated_store, monkeypatch):
    # Driven through asyncio.run rather than pytest.mark.asyncio: pytest-asyncio
    # is not a project dependency, so a coroutine test would be silently skipped.
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

    captured = {}

    async def fake_generate_report(**kwargs):
        context = kwargs["context"]
        assert context["performance"]["headline"]["accuracy"] == 0.8  # noqa: PLR2004
        captured["context"] = context
        return {"markdown": "# Snapshot report\n", "lint": [], "reviewed": True}

    monkeypatch.setattr(analysis, "build_xai", forbidden)
    monkeypatch.setattr(report_agent, "generate_report", fake_generate_report)
    response = asyncio.run(
        analysis.generate_ai_report(
            analysis.ReportRequest(
                optimization_id="run-1",
                analysis_snapshot_id=snapshot["id"],
                analysis_revision=snapshot["revision"],
                api_key="not-persisted",
                llm_provider="openai",
                model_name="test-model",
            )
        )
    )

    assert response["report_markdown"] == "# Snapshot report\n"
    assert analysis_store.list_reports(snapshot["id"])[0]["markdown"] == "# Snapshot report\n"
    # A snapshot whose run record is gone still yields a bundle, with the gap
    # recorded rather than silently dropped.
    assert any("Run configuration" in note for note in captured["context"]["omissions"])


def test_context_and_bundle_endpoints_serve_a_completed_snapshot(isolated_store):
    """The research dump is reachable for any completed snapshot, report or not."""
    image = "data:image/png;base64," + base64.b64encode(b"bundle-png").decode()
    job = analysis_store.create_job("run-2", {})
    analysis_store.complete_job(
        job["id"],
        {
            "metrics": {"accuracy": 0.9},
            "plots": {"bar": image},
            "confusion_matrix_plot": image,
        },
    )
    snapshot = analysis_store.get_snapshot(job["snapshot_id"])

    context = asyncio.run(analysis.get_report_context(snapshot["id"]))
    assert context["context"]["analysis"]["snapshot_id"] == snapshot["id"]
    assert {figure["id"] for figure in context["context"]["figures"]} == {
        "shap_bar",
        "confusion_matrix",
    }
    assert "## Figure manifest" in context["evidence_markdown"]

    response = asyncio.run(analysis.download_research_bundle(snapshot["id"]))
    assert response.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        names = set(archive.namelist())
        assert {"context.json", "evidence.md", "figures/shap_bar.png"} <= names
        assert archive.read("figures/shap_bar.png") == b"bundle-png"


def test_bulk_bundle_download_includes_every_selected_run(isolated_store, monkeypatch):
    for run_id in ("run-1", "run-2"):
        job = analysis_store.create_job(run_id, {})
        analysis_store.complete_job(job["id"], {"metrics": {"accuracy": 0.9}, "plots": {}})

    monkeypatch.setattr(
        analysis,
        "get_job",
        lambda run_id: {
            "id": run_id,
            "status": "failed",
            "request": {"study_name": f"study-{run_id}", "database_name": "results"},
            "trials": [],
        },
    )
    response = asyncio.run(
        analysis.download_bulk_research_bundles(
            analysis.BulkResearchBundleRequest(
                optimization_ids=["run-1", "run-2", "no-analysis", "run-1"]
            )
        )
    )

    assert response.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.body)) as archive:
        bundles = [
            name
            for name in archive.namelist()
            if name.startswith("runs/") and name.endswith(".zip")
        ]
        assert len(bundles) == 3
        bundle_contents = []
        for name in bundles:
            with zipfile.ZipFile(io.BytesIO(archive.read(name))) as bundle:
                bundle_contents.append(set(bundle.namelist()))
        assert any(
            {"README.md", "run.json", "trials.json"} <= contents for contents in bundle_contents
        )
        assert sum("context.json" in contents for contents in bundle_contents) == 2


def test_a_running_analysis_is_findable_without_its_job_id(isolated_store):
    """A refreshed browser loses the job id; the work keeps going without it.

    Without this lookup the UI reports nothing in flight and starts a second
    run over the top of the first.
    """
    config = {"trial_number": 3, "subset_size": 25}
    started = analysis_store.create_job("run-1", config)

    found = analysis_store.find_active_job("run-1")
    assert found is not None
    assert found["id"] == started["id"]
    assert found["snapshot_id"] == started["snapshot_id"]
    assert found["status"] == "pending"
    # The config comes back too, so a reattaching client knows what is running.
    assert found["config"]["trial_number"] == config["trial_number"]


def test_a_finished_analysis_is_not_reported_as_active(isolated_store):
    config = {"trial_number": 3, "subset_size": 25}
    started = analysis_store.create_job("run-1", config)
    analysis_store.complete_job(started["id"], {"metrics": {"f1": 0.9}, "plots": {}})

    assert analysis_store.find_active_job("run-1") is None


def test_active_lookup_is_scoped_to_its_optimization(isolated_store):
    analysis_store.create_job("run-1", {"trial_number": 1})

    assert analysis_store.find_active_job("run-2") is None
    assert asyncio.run(analysis.find_active_analysis_job("run-2")) == {"job": None}
    endpoint = asyncio.run(analysis.find_active_analysis_job("run-1"))
    assert endpoint["job"]["status"] == "pending"
