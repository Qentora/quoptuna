"""Wiring tests for the analyst -> reviewer -> normalise report pipeline.

The LLM calls are stubbed, so these assert what the pipeline does around them:
that both agents receive the full evidence bundle and the figure manifest, that
the reviewer's output wins but an empty reviewer cannot destroy the report, and
that the markdown contract is enforced in code rather than merely requested.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from quoptuna.backend.xai import report_agent, report_context
from quoptuna.backend.xai.prompts import DEFAULT_ANALYST_PROMPT, DEFAULT_REVIEWER_PROMPT

PNG = "data:image/png;base64," + base64.b64encode(b"png").decode()


class _Result:
    def __init__(self, text: str):
        self.final_output = text


@pytest.fixture
def context() -> dict:
    return report_context.build_context(
        optimization_id="opt_1",
        snapshot={
            "id": "snap-1",
            "optimization_id": "opt_1",
            "revision": 1,
            "config": {"trial_number": 0, "class_index": 0, "sample_index": 0},
            "payload": {
                "metrics": {"f1_score": 0.87, "accuracy": 0.9},
                "plots": {"bar": PNG},
                "confusion_matrix_plot": PNG,
                "feature_importance": [{"feature": "age", "importance": 0.4}],
            },
        },
        run={"request": {"study_name": "s", "sampler": "tpe", "num_trials": 3}},
    )


@pytest.fixture
def stub_runner(monkeypatch):
    """Capture each agent call and return scripted output for it."""
    calls: list[dict] = []
    scripted: dict[str, str] = {}

    async def run(agent, payload, **kwargs):
        text = "".join(
            block.get("text", "")
            for message in payload
            for block in message["content"]
            if block["type"] == "input_text"
        )
        images = [
            block["image_url"]
            for message in payload
            for block in message["content"]
            if block["type"] == "input_image"
        ]
        calls.append(
            {
                "agent": agent.name,
                "instructions": agent.instructions,
                "text": text,
                "images": images,
            }
        )
        return _Result(scripted.get(agent.name, "# Fallback\n"))

    # Agent validates its model argument; a plain string is an accepted form and
    # keeps the stub from touching LiteLLM.
    monkeypatch.setattr(report_agent, "_build_model", lambda *a, **k: "stub-model")
    monkeypatch.setattr(report_agent.Runner, "run", staticmethod(run))
    return calls, scripted


def generate(context: dict, **kwargs) -> dict:
    return asyncio.run(
        report_agent.generate_report(
            context=context,
            images=report_context.figure_images(
                {
                    "plots": {"bar": PNG},
                    "confusion_matrix_plot": PNG,
                },
                context["figures"],
            ),
            api_key="k",
            provider="openai",
            model_name="stub",
            **kwargs,
        )
    )


def test_both_agents_get_the_bundle_the_manifest_and_the_figures(context, stub_runner):
    calls, scripted = stub_runner
    scripted["Analyst"] = "# Draft\n\nBody.\n"
    scripted["Reviewer"] = "# Reviewed\n\nBody.\n"

    result = generate(context)

    assert [call["agent"] for call in calls] == ["Analyst", "Reviewer"]
    analyst, reviewer = calls
    assert analyst["instructions"] == DEFAULT_ANALYST_PROMPT
    assert reviewer["instructions"] == DEFAULT_REVIEWER_PROMPT
    for call in calls:
        # The configuration table and the figure manifest reach both agents:
        # the reviewer cannot re-ground numbers it was never shown.
        assert "RUN EVIDENCE" in call["text"]
        assert "FIGURE MANIFEST" in call["text"]
        assert "## Configuration chosen for this run" in call["text"]
        assert "figures/shap_bar.png" in call["text"]
    assert len(analyst["images"]) == 2  # noqa: PLR2004 - bar + confusion matrix
    assert reviewer["images"] == []  # the reviewer re-checks text, not pixels
    assert result["markdown"] == "# Reviewed\n\nBody.\n"
    assert result["reviewed"] is True


def test_review_can_be_skipped(context, stub_runner):
    calls, scripted = stub_runner
    scripted["Analyst"] = "# Draft\n\nBody.\n"

    result = generate(context, enable_review=False)

    assert [call["agent"] for call in calls] == ["Analyst"]
    assert result["markdown"] == "# Draft\n\nBody.\n"
    assert result["reviewed"] is False


def test_custom_instructions_replace_the_defaults(context, stub_runner):
    calls, scripted = stub_runner
    scripted["Analyst"] = scripted["Reviewer"] = "# T\n\nBody.\n"

    generate(context, analyst_instructions="  Be terse.  ", reviewer_instructions="")

    assert calls[0]["instructions"] == "Be terse."
    # A blank override falls back to the built-in prompt rather than no prompt.
    assert calls[1]["instructions"] == DEFAULT_REVIEWER_PROMPT


def test_an_empty_reviewer_cannot_destroy_the_report(context, stub_runner):
    _, scripted = stub_runner
    scripted["Analyst"] = "# Draft\n\n" + "Substantive body. " * 20 + "\n"
    scripted["Reviewer"] = "I cannot help with that."

    result = generate(context)

    assert result["markdown"].startswith("# Draft")


def test_output_is_normalised_and_invented_figures_are_dropped(context, stub_runner):
    _, scripted = stub_runner
    scripted["Analyst"] = scripted["Reviewer"] = (
        "Here is the report:\n\n```markdown\n"
        "Title\n=====\n## S\n| A | B |\n| 1 |\n"
        "![real](figures/shap_bar.png)\n"
        "![fake](figures/not_a_figure.png)\n"
        "```\n"
    )

    result = generate(context)

    markdown = result["markdown"]
    assert markdown.startswith("# Title\n")
    assert "```" not in markdown  # the wrapping fence is gone
    assert "| --- |" in markdown  # a delimiter row was inserted
    assert result["dropped_figures"] == ["not_a_figure"]
    assert result["referenced_figures"] == ["shap_bar"]
    assert result["lint"] == []


def test_unknown_provider_is_rejected_before_any_call(context, monkeypatch):
    monkeypatch.setattr(
        report_agent.Runner, "run", staticmethod(lambda *a, **k: pytest.fail("no call expected"))
    )
    with pytest.raises(ValueError, match="Invalid provider"):
        asyncio.run(
            report_agent.generate_report(
                context=context, api_key="k", provider="nope", model_name="m"
            )
        )
