"""Jobs must not outlive the process that ran them, and must be stoppable.

An analysis is an in-process background task. Its database row survives a
restart while the work does not, so a client asking what is still running would
be handed a job that can never progress — the UI reattaches and sits on
"Computing SHAP" forever.
"""

from __future__ import annotations

import asyncio

import pytest

from quoptuna.server.api.v1 import analysis
from quoptuna.server.services import analysis_store, run_store


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr(analysis_store, "ARTIFACT_ROOT", tmp_path / "analysis")
    return tmp_path


def test_a_job_left_running_by_a_dead_process_is_failed_at_startup(isolated_store):
    started = analysis_store.create_job("run-1", {"trial_number": 1})

    assert analysis_store.abandon_orphaned_jobs() == 1

    job = analysis_store.get_job(started["id"])
    assert job["status"] == "failed"
    assert "restart" in job["error"]
    # ...and it must no longer be offered as something to reattach to.
    assert analysis_store.find_active_job("run-1") is None


def test_reaping_leaves_finished_jobs_alone(isolated_store):
    done = analysis_store.create_job("run-1", {"trial_number": 1})
    analysis_store.complete_job(done["id"], {"metrics": {"f1": 0.9}, "plots": {}})

    assert analysis_store.abandon_orphaned_jobs() == 0
    assert analysis_store.get_job(done["id"])["status"] == "completed"


def test_cancelling_marks_the_job_for_a_stop(isolated_store):
    started = analysis_store.create_job("run-1", {"trial_number": 1})

    result = asyncio.run(analysis.cancel_analysis_job(started["id"]))

    assert result["cancelled"] is True
    assert analysis._is_cancelled(started["id"])
    with pytest.raises(BaseException, match=started["id"]):
        analysis._stop_if_cancelled(started["id"])
    analysis._clear_cancel(started["id"])


def test_cancelling_a_finished_job_is_a_no_op(isolated_store):
    done = analysis_store.create_job("run-1", {"trial_number": 1})
    analysis_store.complete_job(done["id"], {"metrics": {"f1": 0.9}, "plots": {}})

    result = asyncio.run(analysis.cancel_analysis_job(done["id"]))

    assert result["cancelled"] is False
    assert not analysis._is_cancelled(done["id"])


def test_a_cancellation_survives_the_broad_handlers_around_it():
    """It is a BaseException on purpose: `except Exception` must not eat a stop."""

    def stop_behind_a_broad_handler() -> None:
        try:
            analysis._stop_if_cancelled("j-cancel")
        except Exception:  # noqa: BLE001 - mirrors the job body's own handler
            pytest.fail("cancellation was swallowed by an `except Exception`")

    analysis._request_cancel("j-cancel")
    try:
        with pytest.raises(analysis._JobCancelled):
            stop_behind_a_broad_handler()
    finally:
        analysis._clear_cancel("j-cancel")
