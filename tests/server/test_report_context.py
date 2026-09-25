"""Regression tests for the report evidence bundle (the "research dump").

These pin down what the old report path silently dropped: the run configuration,
the trial history, the fairness-aware search settings and the Pareto front. A
regression here means the agent is once again writing about a run it cannot see.
"""

from __future__ import annotations

import base64
import io
import json
import zipfile

import pytest

from quoptuna.backend.xai import markdown_report, report_context
from quoptuna.server.services import research_bundle

PNG = "data:image/png;base64," + base64.b64encode(b"png-bytes").decode()


@pytest.fixture
def payload() -> dict:
    return {
        "metrics": {"f1_score": 0.87, "accuracy": 0.89, "confusion_matrix": [[12, 2], [3, 10]]},
        "feature_importance": [{"feature": "age", "importance": 0.4}],
        "plots": {"bar": PNG, "rocCurve": PNG},
        "confusion_matrix_plot": PNG,
        "confusion_data": {"labels": ["no", "yes"], "matrix": [[12, 2], [3, 10]]},
        "study_plots": {
            "param_importances": {
                "data": [{"x": [0.7, 0.3], "y": ["n_layers", "learning_rate"]}],
                "layout": {},
            },
            "timeline": None,
        },
        "shap_data": {
            "feature_names": ["age"],
            "values": [[0.4], [0.2]],
            "data": [[30.0], [50.0]],
            "base_value": 0.1,
            "n_samples": 2,
        },
        "task_type": "binary",
        "class_labels": ["no", "yes"],
        "fairness": {
            "sensitive_feature": "sex",
            "metrics": {
                "by_group": {
                    "accuracy": {"F": 0.8, "M": 0.95},
                    "count": {"F": 10, "M": 15},
                    "recall": {"F": 0.6, "M": 0.9},
                },
                "overall": {"accuracy": 0.89},
                "disparities": {"equal_opportunity_difference": 0.3, "disparate_impact": 0.46},
            },
            "plots": {"accuracy": PNG},
            "mitigation": None,
        },
        "warnings": {},
    }


@pytest.fixture
def snapshot(payload: dict) -> dict:
    return {
        "id": "snap-1",
        "optimization_id": "opt_1",
        "revision": 2,
        "completed_at": "2026-09-12T10:00:00",
        "storage_backend": "local",
        "config": {
            "trial_number": 1,
            "use_proba": True,
            "subset_size": 50,
            "class_index": 0,
            "sample_index": 0,
        },
        "payload": payload,
    }


@pytest.fixture
def run() -> dict:
    return {
        "status": "completed",
        "started_at": "2026-09-12T09:00:00",
        "completed_at": "2026-09-12T09:30:00",
        "best_value": 0.87,
        "best_params": {"model_type": "SVC", "C": 1},
        "best_trial_number": 1,
        "request": {
            "dataset_id": "2",
            "study_name": "study-a",
            "database_name": "results",
            "num_trials": 5,
            "sampler": "tpe",
            "pruner": "none",
            "resampling": "oversample",
            "categorical_encoding": "onehot",
            "dev_type": "lightning.qubit",
            "model_types": ["SVC", "DataReuploadingClassifier"],
            "sensitive_feature": "sex",
            "fairness_mode": "multi_objective",
            "fairness_metric": "equal_opportunity_difference",
            "fairness_threshold": 0.1,
            "selected_features": ["age"],
            "target_column": "y",
        },
    }


TRIALS = [
    {
        "trial": 0,
        "value": 0.62,
        "values": [0.62, 0.08],
        "params": {"model_type": "SVC", "C": 1},
        "state": "COMPLETE",
        "user_attrs": {"fairness_disparity": 0.08, "training_time": 2.0},
    },
    {
        "trial": 1,
        "value": 0.87,
        "values": [0.87, 0.30],
        "params": {"model_type": "SVC", "C": 10},
        "state": "COMPLETE",
        "user_attrs": {"fairness_disparity": 0.30, "training_time": 3.0},
    },
    {
        "trial": 2,
        "value": None,
        "values": None,
        "params": {"model_type": "DataReuploadingClassifier"},
        "state": "FAIL",
        "user_attrs": {"error": "boom"},
    },
]

PARETO = [
    {"trial": 1, "values": [0.87, 0.30], "params": {"model_type": "SVC", "C": 10}},
    {"trial": 0, "values": [0.62, 0.08], "params": {"model_type": "SVC", "C": 1}},
]


def build(snapshot, run, **kwargs) -> dict:
    return report_context.build_context(
        optimization_id="opt_1",
        snapshot=snapshot,
        run=run,
        dataset={"id": "2", "name": "Adult", "source": "uci", "rows": 100, "columns": ["age", "y"]},
        trials=TRIALS,
        pareto_trials=PARETO,
        **kwargs,
    )


def test_context_is_json_serialisable_and_carries_every_chosen_option(snapshot, run):
    context = build(snapshot, run)
    json.dumps(context)  # must survive the API and the bundle

    search = context["configuration"]["search"]
    assert search["sampler"] == "tpe"
    assert search["model_types"] == ["SVC", "DataReuploadingClassifier"]
    assert context["configuration"]["data"]["resampling"] == "oversample"
    assert context["configuration"]["data"]["categorical_encoding"] == "onehot"
    assert context["configuration"]["training"]["dev_type"] == "lightning.qubit"
    assert context["analysis"]["snapshot_id"] == "snap-1"
    assert context["run"]["duration_seconds"] == 1800  # noqa: PLR2004


def test_trial_history_and_model_family_aggregates_cover_every_trial(snapshot, run):
    context = build(snapshot, run)
    optimization = context["optimization"]
    assert optimization["state_counts"] == {"COMPLETE": 2, "FAIL": 1}
    assert optimization["n_trials_recorded"] == len(TRIALS)
    # Ranked best-first so a row cap keeps the informative rows.
    assert [row["trial"] for row in optimization["trials"]] == [1, 0, 2]
    svc = next(f for f in optimization["model_families"] if f["model_type"] == "SVC")
    assert svc["trials"] == 2  # noqa: PLR2004
    assert svc["best_objective"] == 0.87  # noqa: PLR2004
    assert svc["failed"] == 0


def test_trial_row_cap_is_recorded_not_silent(snapshot, run):
    context = build(snapshot, run, inclusions=report_context.ReportInclusions(max_trial_rows=1))
    assert len(context["optimization"]["trials"]) == 1
    assert context["optimization"]["n_trials_recorded"] == len(TRIALS)
    assert any("row cap" in note for note in context["omissions"])


def test_fairness_aware_search_and_pareto_front_reach_the_bundle(snapshot, run):
    context = build(snapshot, run)
    search = context["fairness"]["search"]
    assert search["mode"] == "multi_objective"
    assert search["metric"] == "equal_opportunity_difference"
    assert search["threshold_effective"] == 0.1  # noqa: PLR2004
    # Per-trial disparities recorded by the optimizer become feasibility evidence.
    assert search["trial_disparities"]["n_trials_scored"] == 2  # noqa: PLR2004
    assert search["trial_disparities"]["n_feasible"] == 1

    pareto = context["pareto_front"]
    assert pareto["present"] is True
    assert pareto["included"] is True
    assert pareto["n_points"] == len(PARETO)
    assert pareto["knee_trial"] in {0, 1}
    assert pareto["reported_trial"] == 1

    rendered = report_context.render_markdown(context)
    assert "## Fairness-aware search" in rendered
    assert "## Pareto front" in rendered
    assert "Pareto-optimal trials" in rendered


def test_disparate_impact_threshold_defaults_to_the_four_fifths_rule(snapshot, run):
    run["request"]["fairness_metric"] = "disparate_impact"
    run["request"]["fairness_threshold"] = None
    context = build(snapshot, run)
    search = context["fairness"]["search"]
    assert search["threshold_effective"] == 0.8  # noqa: PLR2004
    assert "four-fifths" in search["direction"]
    # DI is a ratio; feasibility is compared in disparity space (1 - ratio).
    assert search["trial_disparities"]["feasibility_limit_in_disparity_space"] == 0.2  # noqa: PLR2004


def test_excluding_fairness_records_the_exclusion(snapshot, run):
    context = build(
        snapshot,
        run,
        inclusions=report_context.ReportInclusions(
            fairness=False, fairness_search=False, pareto=False
        ),
    )
    assert context["fairness"]["audit_included"] is False
    assert context["pareto_front"]["included"] is False
    notes = " ".join(context["omissions"])
    assert "excluded by the report settings" in notes
    rendered = report_context.render_markdown(context)
    assert "## Fairness audit" not in rendered
    assert "## Pareto front" not in rendered


def test_no_protected_attribute_tells_the_agent_not_to_speculate(snapshot, run):
    snapshot["payload"] = {**snapshot["payload"], "fairness": None}
    run["request"] = {**run["request"], "sensitive_feature": None, "fairness_mode": "off"}
    context = build(snapshot, run)
    assert any("do not infer group harms" in note for note in context["omissions"])


def test_rendered_evidence_is_valid_markdown_with_a_figure_manifest(snapshot, run):
    context = build(snapshot, run)
    rendered = report_context.render_markdown(context)
    # No level-1 title: the evidence bundle is an input document, not a report.
    assert lint_issues(rendered) == []
    ids = [figure["id"] for figure in context["figures"]]
    assert ids[:1] == ["confusion_matrix"]
    assert "shap_bar" in ids
    assert "roc_curve" in ids
    assert "fairness_accuracy" in ids
    assert "study_param_importances" in ids
    for figure_id in ids:
        assert f"figures/{figure_id}.png" in rendered or f"{figure_id}.json" in rendered


def lint_issues(rendered: str) -> list[str]:
    return [
        issue for issue in markdown_report.lint_markdown(rendered) if "level-1 title" not in issue
    ]


def test_param_importances_are_extracted_from_the_plotly_figure(snapshot, run):
    context = build(snapshot, run)
    importances = context["diagnostics"]["param_importances"]
    assert importances[0] == {"parameter": "n_layers", "importance": 0.7}
    assert any("timeline" in note for note in context["omissions"])


def test_shap_statistics_quantify_direction_not_just_magnitude(snapshot, run):
    context = build(snapshot, run)
    stats = context["explainability"]["shap_statistics"]
    assert stats[0]["feature"] == "age"
    assert stats[0]["mean_abs_shap"] == 0.3  # noqa: PLR2004
    assert stats[0]["feature_min"] == 30  # noqa: PLR2004
    assert stats[0]["feature_max"] == 50  # noqa: PLR2004


def test_bundle_contains_figures_tables_and_the_exact_agent_evidence(snapshot, run, payload):
    context = build(snapshot, run)
    evidence = report_context.render_markdown(context)
    archive = research_bundle.build_zip(
        context=context,
        payload=payload,
        reports=[
            {
                "id": "r1",
                "markdown": "# Report",
                "status": "completed",
                "provider": "google",
                "model_name": "gemini-2.5-pro",
                "created_at": "2026-09-12T10:30:00",
                "snapshot_revision": 2,
            }
        ],
        evidence_markdown=evidence,
        prompts={"analyst": "A", "reviewer": "R"},
        run=run,
        trials=TRIALS,
        pareto_trials=PARETO,
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        names = set(zf.namelist())
        assert {"README.md", "context.json", "evidence.md", "report.md", "run.json"} <= names
        assert "figures/shap_bar.png" in names
        assert "figures/index.md" in names
        assert "study_plots/study_param_importances.json" in names
        assert "tables/pareto_front.csv" in names
        assert "tables/fairness_by_group.csv" in names
        assert "prompts/analyst.md" in names
        assert zf.read("figures/shap_bar.png") == b"png-bytes"
        assert zf.read("evidence.md").decode() == evidence
        # Every figure the report may reference resolves inside the archive.
        exported = json.loads(zf.read("context.json"))["figures"]
        for figure in exported:
            if figure["kind"] == "image":
                assert figure["bundle_path"] in names



def test_bundle_filename_includes_study_and_run_names(snapshot, run):
    filename = research_bundle.bundle_filename(build(snapshot, run))

    assert "study-a-rev" in filename
    assert "opt_1" not in filename

def test_bundle_never_exports_the_live_result_or_a_key(snapshot, run, payload):
    context = build(snapshot, run)
    archive = research_bundle.build_zip(
        context=context,
        payload=payload,
        run={**run, "result": {"x_train": "DATAFRAME"}, "api_key": "secret-key"},
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        run_json = zf.read("run.json").decode()
    assert "secret-key" not in run_json
    assert "DATAFRAME" not in run_json


def test_missing_run_record_degrades_instead_of_failing(snapshot):
    context = report_context.build_context(optimization_id="opt_1", snapshot=snapshot, run=None)
    json.dumps(context)
    assert any("Run configuration" in note for note in context["omissions"])
    assert report_context.render_markdown(context)
