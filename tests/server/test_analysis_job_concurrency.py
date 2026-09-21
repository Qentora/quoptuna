"""The analysis job must not occupy the event loop.

Every step of an analysis — rehydration, the model refit, SHAP, figure
rendering — is synchronous CPU work. Run as an ``async def`` background task it
held the event loop for the whole job, so the server could not answer a single
``GET /analysis/jobs/{id}`` poll: the UI showed "Starting analysis" and then
jumped straight to whichever section was current once the loop came back, which
was always SHAP. Per-row SHAP progress was unreadable for the same reason.
"""

from __future__ import annotations

import asyncio
import inspect
import threading
import time

from starlette.concurrency import run_in_threadpool

from quoptuna.server.api.v1 import analysis

JOB_SECONDS = 0.4
POLL_INTERVAL = 0.02
# Well under JOB_SECONDS / POLL_INTERVAL, so timing jitter cannot fail this.
MIN_POLLS = 5
WORKERS = 3


def _request() -> analysis.AnalysisJobRequest:
    return analysis.AnalysisJobRequest(optimization_id="run-1")


def test_the_job_is_not_a_coroutine_function():
    """This is what makes Starlette run it in the threadpool rather than the loop."""
    assert not inspect.iscoroutinefunction(analysis._run_analysis_job)


def test_polls_are_answered_while_a_job_runs(monkeypatch):
    async def busy(job_id: str, request) -> None:
        time.sleep(JOB_SECONDS)  # noqa: ASYNC251 - standing in for CPU-bound work

    monkeypatch.setattr(analysis, "_run_analysis_job_async", busy)

    polls = 0

    async def scenario() -> int:
        nonlocal polls
        job = asyncio.ensure_future(run_in_threadpool(analysis._run_analysis_job, "j1", _request()))
        while not job.done():
            polls += 1
            await asyncio.sleep(POLL_INTERVAL)
        await job
        return polls

    assert asyncio.run(scenario()) >= MIN_POLLS


def test_jobs_do_not_overlap(monkeypatch):
    """Figures render through pyplot's global state, so keep jobs serialized."""
    concurrent = 0
    peak = 0
    guard = threading.Lock()

    async def busy(job_id: str, request) -> None:
        nonlocal concurrent, peak
        with guard:
            concurrent += 1
            peak = max(peak, concurrent)
        time.sleep(0.05)  # noqa: ASYNC251 - standing in for CPU-bound work
        with guard:
            concurrent -= 1

    monkeypatch.setattr(analysis, "_run_analysis_job_async", busy)

    async def scenario() -> None:
        await asyncio.gather(
            *(
                run_in_threadpool(analysis._run_analysis_job, f"j{i}", _request())
                for i in range(WORKERS)
            )
        )

    asyncio.run(scenario())
    assert peak == 1
