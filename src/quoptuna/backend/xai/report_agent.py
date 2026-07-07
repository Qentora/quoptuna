"""LLM report generation on the OpenAI Agents SDK.

A two-agent pipeline replaces the old single LangChain call:

1. ``Analyst`` — receives the metrics report plus every plot (base64 data
   URLs) and drafts the full markdown report from ``prompt.txt``.
2. ``Reviewer`` — receives the draft plus the metrics text only and removes
   hallucinated numbers, tightens structure, and returns the final markdown.

All providers are routed through LiteLLM (``LitellmModel``) so the existing
google/openai selection — and now anthropic — keep working with per-request
API keys (no environment mutation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from agents import Agent, Runner, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel

# LiteLLM provider prefixes for the providers exposed by the app.
_PROVIDER_PREFIXES = {
    "openai": "openai",
    "google": "gemini",
    "anthropic": "anthropic",
}

_REVIEWER_INSTRUCTIONS = """You are a meticulous technical editor reviewing a
model-evaluation report before it is delivered.

You are given the draft report and the raw metrics it must be grounded in.
Your job:
1. Verify every number cited in the draft appears in (or is directly
   derivable from) the raw metrics; correct or remove any that do not.
2. Do not invent new metrics, groups, or findings.
3. Tighten structure and wording; keep all section headings.
4. Preserve markdown formatting.

Output ONLY the final markdown report, with no preamble or commentary."""


def _build_model(provider: str, model_name: str, api_key: str) -> LitellmModel:
    prefix = _PROVIDER_PREFIXES.get(provider)
    if prefix is None:
        msg = f"Invalid provider: {provider!r} (expected one of {sorted(_PROVIDER_PREFIXES)})"
        raise ValueError(msg)
    return LitellmModel(model=f"{prefix}/{model_name}", api_key=api_key)


def _analyst_input(report_text: str, images: dict[str, str]) -> list[dict]:
    content: list[dict] = [
        {"type": "input_text", "text": f"Model Evaluation Report:\n```\n{report_text}\n```"}
    ]
    for plot_type, image_url in images.items():
        content.append(
            {"type": "input_text", "text": f"Here is a {plot_type.replace('_', ' ')} plot:"}
        )
        content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
    return [{"role": "user", "content": content}]


def load_system_prompt() -> str:
    """Load the analyst system prompt bundled next to this module."""
    return (Path(__file__).parent / "prompt.txt").read_text()


async def generate_report(
    report: dict,
    images: dict[str, str],
    api_key: str,
    model_name: str = "gpt-4o",
    provider: str = "google",
) -> str:
    """Draft + review a markdown report grounded in the given metrics and plots."""
    # The SDK's tracing exporter needs an OpenAI key; requests may use other
    # providers (or per-request keys), so disable it outright.
    set_tracing_disabled(disabled=True)

    model = _build_model(provider, model_name, api_key)
    report_text = str(report)

    analyst = Agent(
        name="Analyst",
        instructions=load_system_prompt(),
        model=model,
    )
    draft = await Runner.run(analyst, cast("Any", _analyst_input(report_text, images)))

    reviewer = Agent(
        name="Reviewer",
        instructions=_REVIEWER_INSTRUCTIONS,
        model=model,
    )
    reviewed = await Runner.run(
        reviewer,
        cast(
            "Any",
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": f"Draft report:\n\n{draft.final_output}"},
                        {
                            "type": "input_text",
                            "text": (
                                f"Raw metrics the report must be grounded in:\n```\n{report_text}\n```"
                            ),
                        },
                    ],
                }
            ],
        ),
    )
    return str(reviewed.final_output)
