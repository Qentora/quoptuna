"""One-click research dump for an analysis snapshot.

Everything an agent was given - and everything it was given it from - packaged as
a single zip: the structured context (``context.json``), the exact markdown the
agents read (``evidence.md``), every figure as a real file under ``figures/``,
the Optuna study figures as Plotly JSON, flat CSV tables, the prompts used, and
the generated reports. A report's ``figures/shap_bar.png`` reference resolves
inside the archive, so a downloaded bundle renders as-is in any markdown viewer.

Figures live in the snapshot payload as base64 data URLs (local storage) or as
presigned object-store URLs (S3). The first are written into the archive; the
second are recorded as links in ``figures/index.md`` and on each manifest entry,
because re-fetching them here would tie the download to object-store reachability.
"""

from __future__ import annotations

import base64
import csv
import io
import json
import re
import zipfile
from datetime import datetime
from typing import Any, Iterable, Sequence

_DATA_URL = re.compile(r"^data:(?P<mime>[\w/.+-]+);base64,(?P<data>.*)$", re.DOTALL)
_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/svg+xml": "svg"}
_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]+")


def _resolve(payload: dict, dotted: str) -> Any:
    """Follow a ``figures[*].source`` path (e.g. ``fairness.plots.accuracy``)."""
    current: Any = payload
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def resolve_figure_assets(payload: dict, figures: Sequence[dict]) -> dict[str, dict[str, Any]]:
    """Map each manifest figure id to its bytes, link, or Plotly JSON."""
    assets: dict[str, dict[str, Any]] = {}
    for figure in figures:
        source = figure.get("source") or ""
        value = _resolve(payload, source) if source else None
        if value is None:
            continue
        if figure.get("kind") == "plotly":
            assets[figure["id"]] = {
                "kind": "plotly",
                "path": f"study_plots/{figure['id']}.json",
                "json": value,
            }
            continue
        if not isinstance(value, str):
            continue
        match = _DATA_URL.match(value.strip())
        if match:
            mime = match.group("mime")
            extension = _EXTENSIONS.get(mime, "png")
            try:
                raw = base64.b64decode(match.group("data"))
            except (ValueError, TypeError):
                continue
            assets[figure["id"]] = {
                "kind": "image",
                "mime": mime,
                "path": f"figures/{figure['id']}.{extension}",
                "bytes": raw,
            }
        elif value.startswith(("http://", "https://")):
            assets[figure["id"]] = {
                "kind": "link",
                "path": f"figures/{figure['id']}.png",
                "url": value,
            }
    return assets


def figure_urls(assets: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Object-store links, keyed by figure id (empty for local storage)."""
    return {fid: asset["url"] for fid, asset in assets.items() if asset.get("url")}


def image_map(payload: dict, figures: Sequence[dict]) -> dict[str, str]:
    """Figure id -> data URL (or presigned URL), for attaching to the agent."""
    from quoptuna.backend.xai.report_context import figure_images

    return figure_images(payload, figures)


def _csv_text(rows: Iterable[Sequence[Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    for row in rows:
        writer.writerow(["" if cell is None else cell for cell in row])
    return buffer.getvalue()


def _safe(name: str) -> str:
    return _SAFE_NAME.sub("_", str(name)).strip("_") or "item"


def _readme(context: dict, assets: dict[str, dict[str, Any]], reports: Sequence[dict]) -> str:
    run = context.get("run") or {}
    analysis = context.get("analysis") or {}
    figures = context.get("figures") or []
    lines = [
        "# QuOptuna research dump",
        "",
        f"Exported {datetime.now().isoformat(timespec='seconds')} from analysis snapshot "
        f"`{analysis.get('snapshot_id')}` revision {analysis.get('revision')} of run "
        f"`{run.get('optimization_id')}`.",
        "",
        "This archive is the complete input/output record of the AI report for this run:",
        "everything the agents were given, in the form they were given it, plus the",
        "artifacts referenced by the generated report.",
        "",
        "## Contents",
        "",
        "| Path | What it is |",
        "| --- | --- |",
        "| `context.json` | The full structured evidence bundle (stable schema). |",
        "| `evidence.md` | The exact markdown rendering of `context.json` sent to the analyst. |",
        "| `figures/` | Every rendered figure. Report references (`figures/<id>.png`) resolve here. |",
        "| `figures/index.md` | Figure manifest: id, title, group, what each plot shows. |",
        "| `study_plots/` | Optuna study figures as Plotly JSON (re-openable with plotly.io). |",
        "| `tables/` | Flat CSV exports of every table in the bundle. |",
        "| `prompts/` | The built-in analyst and reviewer prompts. A prompt customised in "
        "Settings lives in the browser and is not stored server-side; "
        "`context.json` records whether a report used one. |",
        "| `reports/` | Every generated report for this snapshot revision, newest first. |",
        "| `report.md` | The most recent completed report (same file as the newest in `reports/`). |",
        "| `run.json` | The raw optimization request, trial history and Pareto front. |",
        "",
        f"## Figures ({len(figures)})",
        "",
    ]
    if figures:
        lines += ["| Figure | Title | In this archive |", "| --- | --- | --- |"]
        for figure in figures:
            asset = assets.get(figure["id"])
            if asset is None:
                where = "not exported (no stored image)"
            elif asset["kind"] == "link":
                where = f"[object storage]({asset['url']})"
            else:
                where = f"`{asset['path']}`"
            lines.append(f"| `{figure['id']}` | {figure['title']} | {where} |")
    else:
        lines.append("_No figures were recorded for this snapshot._")

    omissions = context.get("omissions") or []
    lines += ["", "## Known gaps in this run", ""]
    lines += [f"- {item}" for item in omissions] or ["- None recorded."]
    lines += ["", f"## Reports ({len(reports)})", ""]
    if reports:
        lines += [
            "| File | Status | Provider | Model | Created |",
            "| --- | --- | --- | --- | --- |",
        ]
        for report in reports:
            lines.append(
                f"| `reports/{_safe(report.get('created_at') or report['id'])}.md` "
                f"| {report.get('status')} | {report.get('provider')} "
                f"| {report.get('model_name')} | {report.get('created_at')} |"
            )
    else:
        lines.append("_No reports have been generated for this snapshot yet._")
    return "\n".join(lines) + "\n"


def build_zip(
    *,
    context: dict,
    payload: dict,
    reports: Sequence[dict] = (),
    evidence_markdown: str | None = None,
    prompts: dict[str, str] | None = None,
    run: dict | None = None,
    trials: Sequence[dict] | None = None,
    pareto_trials: Sequence[dict] | None = None,
) -> bytes:
    """Build the research-dump archive and return its bytes."""
    from quoptuna.backend.xai import report_context

    figures = list(context.get("figures") or [])
    assets = resolve_figure_assets(payload, figures)

    # Record where each figure ended up so context.json is self-describing.
    enriched = []
    for figure in figures:
        asset = assets.get(figure["id"]) or {}
        enriched.append(
            {
                **figure,
                "bundle_path": asset.get("path"),
                "url": asset.get("url"),
                "exported": bool(asset),
            }
        )
    context = {**context, "figures": enriched}
    evidence = evidence_markdown or report_context.render_markdown(context)

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", _readme(context, assets, reports))
        archive.writestr("context.json", json.dumps(context, indent=2, default=str))
        archive.writestr("evidence.md", evidence)
        archive.writestr(
            "figures/index.md", report_context.figure_manifest_markdown(enriched) + "\n"
        )
        for figure in figures:
            asset = assets.get(figure["id"])
            if not asset:
                continue
            if asset["kind"] == "image":
                archive.writestr(asset["path"], asset["bytes"])
            elif asset["kind"] == "plotly":
                archive.writestr(asset["path"], json.dumps(asset["json"], indent=2, default=str))

        for name, rows in report_context.context_tables(context).items():
            archive.writestr(f"tables/{_safe(name)}.csv", _csv_text(rows))

        for name, text in (prompts or {}).items():
            if text:
                archive.writestr(f"prompts/{_safe(name)}.md", text.rstrip() + "\n")

        completed = [r for r in reports if r.get("markdown")]
        for report in completed:
            stem = _safe(report.get("created_at") or report.get("id") or "report")
            archive.writestr(f"reports/{stem}.md", str(report["markdown"]).rstrip() + "\n")
        if completed:
            archive.writestr("report.md", str(completed[0]["markdown"]).rstrip() + "\n")

        archive.writestr(
            "run.json",
            json.dumps(
                {
                    "run": {
                        key: value
                        for key, value in (run or {}).items()
                        # "result" holds live DataFrames/models; "api_key" must
                        # never reach an exported artifact.
                        if key not in ("result", "api_key", "trials", "pareto_trials")
                    },
                    "trials": list(trials or []),
                    "pareto_trials": list(pareto_trials or []),
                },
                indent=2,
                default=str,
            ),
        )
    return buffer.getvalue()


def build_run_data_zip(
    *, run: dict[str, Any], trials: Sequence[dict], pareto_trials: Sequence[dict]
) -> bytes:
    """Archive durable run data when no completed analysis snapshot exists."""
    request = run.get("request") or {}
    study_name = run.get("study_name") or request.get("study_name") or "unnamed-study"
    public_run = {
        key: value
        for key, value in run.items()
        if key not in ("api_key", "result", "trials", "pareto_trials")
    }
    readme = "\n".join(
        [
            f"# QuOptuna run export: {study_name}",
            "",
            "No completed analysis snapshot exists for this run.",
            "This archive contains durable run metadata and its available trial history.",
            "",
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", readme)
        archive.writestr("run.json", json.dumps(public_run, indent=2, default=str))
        archive.writestr("trials.json", json.dumps(list(trials), indent=2, default=str))
        if pareto_trials:
            archive.writestr(
                "pareto_trials.json", json.dumps(list(pareto_trials), indent=2, default=str)
            )
    return buffer.getvalue()


def run_data_filename(run: dict[str, Any]) -> str:
    """Filename for a durable run-data export without analysis artifacts."""
    request = run.get("request") or {}
    study_name = run.get("study_name") or request.get("study_name") or "unnamed-study"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"quoptuna-run-{_safe(study_name)}-{stamp}.zip"


def bundle_filename(context: dict) -> str:
    run = context.get("run") or {}
    study_name = run.get("study_name") or "unnamed-study"
    label = _safe(study_name)
    analysis = context.get("analysis") or {}
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"quoptuna-research-dump-{label}-rev{analysis.get('revision') or 0}-{stamp}.zip"
