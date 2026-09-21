"""LLM report generation on the OpenAI Agents SDK.

A two-agent pipeline with a deterministic third stage:

1. ``Analyst`` - receives the full evidence bundle rendered by
   :mod:`quoptuna.backend.xai.report_context` (configuration, search outcome,
   Pareto front, metrics, SHAP, fairness audit), the figure manifest, and the
   figures themselves, and drafts the report.
2. ``Reviewer`` - receives the draft plus the *same* evidence bundle, strips
   ungrounded numbers, adds sections the draft skipped, and repairs markdown.
3. :mod:`quoptuna.backend.xai.markdown_report` - normalises the result to the
   markdown contract (fences, headings, table geometry, figure references) and
   lints what it could not repair. Prompting alone never gets this right, and a
   report destined for a paper has to parse.

Both agent prompts are overridable per request (the Settings page exposes them),
falling back to :mod:`quoptuna.backend.xai.prompts`.

All providers are routed through LiteLLM (``LitellmModel``) so the existing
google/openai selection - and anthropic - keep working with per-request API keys
(no environment mutation).

``prompt.txt`` next to this module is legacy: it is what the pre-bundle single
agent used and is kept only for the notebooks under ``experiments/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

from quoptuna.backend.xai import markdown_report, report_context
from quoptuna.backend.xai.prompts import DEFAULT_ANALYST_PROMPT, DEFAULT_REVIEWER_PROMPT

logger = logging.getLogger(__name__)

# LiteLLM provider prefixes for the providers exposed by the app.
_PROVIDER_PREFIXES = {
    "openai": "openai",
    "google": "gemini",
    "anthropic": "anthropic",
}

#: Hard cap on attached figures. Every provider has a per-request image limit,
#: and a run with per-class curves plus seven fairness plots can exceed it.
MAX_ATTACHED_FIGURES = 24


def _build_model(provider: str, model_name: str, api_key: str) -> LitellmModel:
    prefix = _PROVIDER_PREFIXES.get(provider)
    if prefix is None:
        msg = f"Invalid provider: {provider!r} (expected one of {sorted(_PROVIDER_PREFIXES)})"
        raise ValueError(msg)
    return LitellmModel(model=f"{prefix}/{model_name}", api_key=api_key)


def load_system_prompt() -> str:
    """The analyst system prompt (now maintained in :mod:`prompts`)."""
    return DEFAULT_ANALYST_PROMPT


def legacy_prompt_file() -> Path:
    """Path to the pre-bundle prompt kept for the experiment notebooks."""
    return Path(__file__).parent / "prompt.txt"


def _attachable(images: dict[str, str], figures: list[dict]) -> list[tuple[dict, str]]:
    """Pair manifest entries with their image data, in manifest order.

    Plotly (study) figures have no raster form, so they are listed in the
    manifest and shipped in the download bundle but never attached here.
    """
    pairs: list[tuple[dict, str]] = []
    for figure in figures:
        if figure.get("kind") != "image":
            continue
        value = images.get(figure["id"])
        if isinstance(value, str) and value:
            pairs.append((figure, value))
    return pairs[:MAX_ATTACHED_FIGURES]


def _analyst_input(evidence: str, manifest: str, attachments: list[tuple[dict, str]]) -> list[dict]:
    content: list[dict] = [
        {
            "type": "input_text",
            "text": (
                "RUN EVIDENCE - the only facts you may use. Every table below is "
                "authoritative; anything absent from it is unavailable.\n\n" + evidence
            ),
        },
        {
            "type": "input_text",
            "text": (
                "FIGURE MANIFEST - reference figures only by these ids and paths.\n\n" + manifest
            ),
        },
    ]
    for figure, image_url in attachments:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Figure `{figure['id']}` - {figure['title']} "
                    f"(reference it as figures/{figure['id']}.png)"
                ),
            }
        )
        content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
    return [{"role": "user", "content": content}]


def _reviewer_input(draft: str, evidence: str, manifest: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "DRAFT REPORT to correct:\n\n" + draft},
                {
                    "type": "input_text",
                    "text": (
                        "RUN EVIDENCE the report must be grounded in - correct or delete "
                        "anything the draft states that this does not support.\n\n" + evidence
                    ),
                },
                {
                    "type": "input_text",
                    "text": "FIGURE MANIFEST - the only valid figure references.\n\n" + manifest,
                },
            ],
        }
    ]


async def generate_report(  # noqa: PLR0913
    *,
    context: dict,
    images: dict[str, str] | None = None,
    api_key: str,
    model_name: str = "gpt-4o",
    provider: str = "google",
    analyst_instructions: str | None = None,
    reviewer_instructions: str | None = None,
    enable_review: bool = True,
) -> dict[str, Any]:
    """Draft, review and normalise a markdown report for one evidence bundle.

    ``context`` is a :func:`report_context.build_context` bundle. Returns the
    final markdown alongside the draft, the rendered evidence (so the exact
    agent input can be shipped in the download bundle), any lint findings, and
    figure ids the agents invented and that were therefore removed.
    """
    # The SDK's tracing exporter needs an OpenAI key; requests may use other
    # providers (or per-request keys), so disable it outright.
    set_tracing_disabled(disabled=True)

    figures = list(context.get("figures") or [])
    evidence = report_context.render_markdown(context)
    manifest = report_context.figure_manifest_markdown(figures)
    attachments = _attachable(images or {}, figures)
    # Only figures the model can actually see (or a Plotly figure it was told
    # about) may be referenced; anything else is dropped in post-processing.
    referenceable = [figure["id"] for figure in figures]

    model = _build_model(provider, model_name, api_key)

    analyst = Agent(
        name="Analyst",
        instructions=(analyst_instructions or "").strip() or DEFAULT_ANALYST_PROMPT,
        model=model,
    )
    draft_run = await Runner.run(
        analyst, cast("Any", _analyst_input(evidence, manifest, attachments))
    )
    draft = _as_text(draft_run.final_output)

    final = draft
    if enable_review:
        reviewer = Agent(
            name="Reviewer",
            instructions=(reviewer_instructions or "").strip() or DEFAULT_REVIEWER_PROMPT,
            model=model,
        )
        review_run = await Runner.run(
            reviewer, cast("Any", _reviewer_input(draft, evidence, manifest))
        )
        reviewed = _as_text(review_run.final_output)
        # A reviewer that returns almost nothing has refused or errored; the
        # draft is a better deliverable than an empty document.
        if len(reviewed.strip()) >= len(draft.strip()) // 2:
            final = reviewed
        else:
            logger.warning(
                "Reviewer output was suspiciously short (%d vs %d chars); keeping the draft",
                len(reviewed.strip()),
                len(draft.strip()),
            )

    markdown = markdown_report.normalize_markdown(final)
    markdown, dropped = markdown_report.rewrite_figure_links(markdown, referenceable)
    if dropped:
        markdown = markdown_report.normalize_markdown(markdown)
    lint = markdown_report.lint_markdown(markdown, referenceable)

    return {
        "markdown": markdown,
        "draft_markdown": markdown_report.normalize_markdown(draft),
        "evidence_markdown": evidence,
        "figure_manifest_markdown": manifest,
        "referenced_figures": markdown_report.referenced_figures(markdown),
        "attached_figures": [figure["id"] for figure, _ in attachments],
        "dropped_figures": dropped,
        "lint": lint,
        "reviewed": bool(enable_review),
    }


def _as_text(output: Any) -> str:
    """Coerce an agent result to text.

    Some providers return structured content blocks rather than a string;
    rendering those with ``str()`` would leak Python reprs into the report.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts = []
        for item in output:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    text = getattr(output, "text", None)
    return text if isinstance(text, str) else str(output)
