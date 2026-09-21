"""Retraining fidelity and analysis history.

Optuna records only what it sampled, so the search-time training budget
(``max_steps`` / ``convergence_interval`` / ``dev_type``) is absent from
``trial.params``. Rebuilding a trial without it silently retrains at the
model class defaults, which is a different model from the one that won.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from quoptuna.backend.models import create_model
from quoptuna.backend.xai import report_context as rc
from quoptuna.server.services.workflow_service import _training_budget

SEARCH_MAX_STEPS = 200
SEARCH_CONVERGENCE_INTERVAL = 100
BEST_TRIAL = 7
ANALYSED_TRIAL = 3
FIRST_REVISION_F1 = 0.9
LEGACY_CONFIG_TRIAL = 5
PARTIAL_F1 = 0.81


def test_search_time_training_budget_is_replayed_not_defaulted():
    opt_result = {
        "max_steps": 200,
        "convergence_interval": 100,
        "dev_type": "lightning.qubit",
    }
    budget = _training_budget(opt_result)

    default = create_model("CircuitCentricClassifier", n_classes=2)
    rebuilt = create_model("CircuitCentricClassifier", n_classes=2, **budget)

    assert default.max_steps != SEARCH_MAX_STEPS, "test is vacuous if the default already matches"
    assert rebuilt.max_steps == SEARCH_MAX_STEPS
    assert rebuilt.convergence_interval == SEARCH_CONVERGENCE_INTERVAL


def test_absent_budget_leaves_model_defaults_untouched():
    # Runs predating these knobs must keep the class defaults rather than
    # having None forced onto them.
    assert _training_budget({}) == {}
    assert _training_budget({"max_steps": None, "dev_type": None}) == {}

    default = create_model("CircuitCentricClassifier", n_classes=2)
    rebuilt = create_model("CircuitCentricClassifier", n_classes=2, **_training_budget({}))
    assert rebuilt.max_steps == default.max_steps


def _snapshot(analysed: dict | None, revision: int = 3, config_trial=None) -> dict:
    payload: dict = {"metrics": {"f1": 0.9}}
    if analysed is not None:
        payload["analysed_model"] = analysed
    return {"revision": revision, "config": {"trial_number": config_trial}, "payload": payload}


def test_report_names_the_model_it_actually_explains():
    snapshot = _snapshot(
        {
            "trial_number": 7,
            "model_type": "DataReuploadingClassifier",
            "selected_by": "best_trial",
            "training_budget": {"max_steps": 200},
        }
    )
    context = rc.build_context(
        optimization_id="r1",
        snapshot=snapshot,
        run={"best_trial_number": 7, "best_value": 0.88, "best_params": {}},
    )
    best = context["optimization"]["best_trial"]
    assert best["analysed_model_type"] == "DataReuploadingClassifier"
    assert best["analysis_revision"] == 3  # noqa: PLR2004

    markdown = rc.render_markdown(context)
    assert "DataReuploadingClassifier" in markdown
    assert "max_steps=200" in markdown


def test_report_flags_an_analysis_of_a_non_winning_trial():
    snapshot = _snapshot(
        {"trial_number": 3, "model_type": "SVC", "selected_by": "explicit", "training_budget": {}}
    )
    context = rc.build_context(
        optimization_id="r1",
        snapshot=snapshot,
        run={"best_trial_number": 7, "best_value": 0.88, "best_params": {}},
    )
    assert context["optimization"]["best_trial"]["analysed_is_best"] is False
    assert "NOT the best trial (best is #7)" in rc.render_markdown(context)


def test_snapshots_without_provenance_still_render():
    # Snapshots written before analysed_model existed fall back to the
    # request config rather than failing.
    context = rc.build_context(
        optimization_id="r1",
        snapshot=_snapshot(None, config_trial=5),
        run={"best_trial_number": 7},
    )
    assert context["optimization"]["best_trial"]["analysed_trial"] == LEGACY_CONFIG_TRIAL
    assert "describe a **" not in rc.render_markdown(context)


@pytest.fixture
def store(monkeypatch):
    """The analysis store, with a small history limit to exercise pruning.

    Database and artifact isolation come from the autouse
    ``isolate_app_database`` fixture in conftest.
    """
    from quoptuna.server.core.config import settings  # noqa: PLC0415
    from quoptuna.server.services import analysis_store  # noqa: PLC0415

    monkeypatch.setattr(settings, "ANALYSIS_HISTORY_LIMIT", 3)
    return analysis_store


def test_every_analysis_run_is_kept_in_history(store):
    config = {"trial_number": None, "use_proba": True, "subset_size": 50}
    for i in range(4):
        job = store.create_job("run-hist", config)
        if not job["created"]:
            store.update_job(job["id"], status="running")
        store.complete_job(
            job["id"],
            {
                "metrics": {"f1": 0.9 + i / 100},
                "analysed_model": {"trial_number": 7, "model_type": "SVC"},
            },
        )

    revisions = store.list_revisions(job["snapshot_id"])
    assert [r["revision"] for r in revisions] == [4, 3, 2, 1]
    assert revisions[0]["analysed_trial"] == BEST_TRIAL
    assert revisions[0]["analysed_model_type"] == "SVC"
    # Superseded revisions stay readable even once their figures are pruned.
    first = store.get_revision(job["snapshot_id"], 1)
    assert first["payload"]["metrics"]["f1"] == FIRST_REVISION_F1


def test_history_beyond_the_limit_keeps_metadata_but_drops_artifacts(store):
    config = {"trial_number": None, "use_proba": True, "subset_size": 50}
    for _ in range(4):
        job = store.create_job("run-prune", config)
        if not job["created"]:
            store.update_job(job["id"], status="running")
        store.complete_job(job["id"], {"metrics": {"f1": 0.9}})

    revisions = {r["revision"]: r for r in store.list_revisions(job["snapshot_id"])}
    assert revisions[4]["artifacts_pruned"] is False
    assert revisions[1]["artifacts_pruned"] is True
    assert not Path(store.get_revision(job["snapshot_id"], 1)["artifact_dir"]).exists()
    # The newest revision's figures survive.
    assert Path(store.get_revision(job["snapshot_id"], 4)["artifact_dir"]).exists()


def test_core_sections_are_published_before_the_job_finishes(store):
    """A running job exposes SHAP/metrics so the UI need not wait for the rest."""
    job = store.create_job("run-partial", {"trial_number": None, "subset_size": 50})
    assert store.get_job(job["id"])["partial"] is None

    store.publish_partial(
        job["id"], {"metrics": {"f1": 0.81}, "plots": {"bar": "data:image/png;base64,AAA"}}
    )

    running = store.get_job(job["id"])
    assert running["status"] == "pending"
    assert running["partial"]["metrics"]["f1"] == PARTIAL_F1
    assert "bar" in running["partial"]["plots"]


def test_warming_caches_tolerates_a_model_without_predict_proba():
    from quoptuna.server.api.v1.analysis import _warm_xai_caches  # noqa: PLC0415

    class OnlyPredictions:
        def __init__(self):
            self.calls = []

        @property
        def shap_values(self):
            self.calls.append("shap_values")
            return [1, 2]

        @property
        def predictions(self):
            self.calls.append("predictions")
            return [0, 1]

        @property
        def predictions_proba(self):
            msg = "no predict_proba"
            raise AttributeError(msg)

    xai = OnlyPredictions()
    _warm_xai_caches(xai, use_proba=True)  # must not raise
    assert "shap_values" in xai.calls
    assert "predictions" in xai.calls


def test_warming_skips_probabilities_when_not_requested():
    from quoptuna.server.api.v1.analysis import _warm_xai_caches  # noqa: PLC0415

    class Tracked:
        def __init__(self):
            self.calls = []

        @property
        def shap_values(self):
            self.calls.append("shap_values")
            return []

        @property
        def predictions(self):
            self.calls.append("predictions")
            return []

        @property
        def predictions_proba(self):
            self.calls.append("predictions_proba")
            return []

    xai = Tracked()
    _warm_xai_caches(xai, use_proba=False)
    assert "predictions_proba" not in xai.calls


@pytest.mark.asyncio
async def test_one_failing_section_does_not_cancel_the_others():
    """gather(return_exceptions=True) must isolate per-section failures."""
    import asyncio  # noqa: PLC0415

    from fastapi import HTTPException  # noqa: PLC0415

    warnings: dict[str, str] = {}

    async def ok(value):
        await asyncio.sleep(0)
        return value

    async def boom():
        raise HTTPException(status_code=500, detail="curves exploded")

    calls = {"curves": boom(), "confusion": ok("cm"), "importance": ok("imp")}
    names = list(calls)
    settled = await asyncio.gather(*(calls[n] for n in names), return_exceptions=True)

    results = {}
    for name, outcome in zip(names, settled, strict=True):
        if isinstance(outcome, BaseException):
            warnings[name] = outcome.detail if isinstance(outcome, HTTPException) else str(outcome)
            results[name] = None
        else:
            results[name] = outcome

    assert results["confusion"] == "cm"
    assert results["importance"] == "imp"
    assert results["curves"] is None
    assert warnings == {"curves": "curves exploded"}


def test_job_reports_preparation_and_training_before_shap(store):
    """The pre-SHAP steps must not report as 'shap'.

    Rehydration and the model refit can dominate the analysis, so reporting
    them as "shap" made a working job look stuck.
    """
    job = store.create_job("run-sections", {"trial_number": None, "subset_size": 50})
    seen = []
    for section in ("preparing", "training", "shap"):
        store.update_job(job["id"], current_section=section)
        seen.append(store.get_job(job["id"])["current_section"])
    assert seen == ["preparing", "training", "shap"]
