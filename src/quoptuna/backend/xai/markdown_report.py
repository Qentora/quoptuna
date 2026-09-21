"""Deterministic markdown normalisation and linting for generated reports.

LLMs drift from a markdown contract no matter how it is worded: they wrap the
document in a fence, emit ragged tables, glue headings to paragraphs, or use
Unicode bullets. Prompting reduces that; it never removes it. So the contract is
*also* enforced here, in code, after the agents have run - which is what makes a
generated report safe to commit to a paper or a repository.

The normaliser is intentionally conservative: it repairs structure (fences,
headings, blank lines, table geometry) and never rewrites prose. Anything it
cannot repair is reported by :func:`lint_markdown` so the API can surface it
instead of silently shipping a broken document.
"""

from __future__ import annotations

import re
from typing import Iterable

# A table row candidate: contains an unescaped pipe and is not a fenced line.
_PIPE = re.compile(r"(?<!\\)\|")
_DELIMITER_CELL = re.compile(r"^:?-{1,}:?$")
_ATX = re.compile(r"^(#{1,})[ \t]*(.*?)[ \t]*#*\s*$")
_FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})(.*)$")
_SETEXT_H1 = re.compile(r"^=+\s*$")
_SETEXT_H2 = re.compile(r"^-{2,}\s*$")
_BULLET = re.compile(r"^(\s*)[•‣▪●·⁃∙](\s+)")  # noqa: RUF001 - these are the glyphs to replace
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d{1,9}[.)])\s+")
_PREAMBLE = re.compile(
    r"^\s*(here(\s+is|'s)|below\s+is|sure[,.!]|certainly[,.!]|of\s+course[,.!]|"
    r"i(\s+have|'ve)\s+(written|drafted|prepared)|as\s+requested)",
    re.IGNORECASE,
)
_IMAGE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<path>[^)\s]+)(?P<title>\s+\"[^\"]*\")?\)")
_FIGURE_PATH = re.compile(r"^(?:\./)?figures/(?P<fid>[A-Za-z0-9_.-]+)\.(?:png|jpg|jpeg|svg)$")
_TRAILING_WS = re.compile(r"[ \t]+$")

_MAX_HEADING_LEVEL = 6
_TABLE_MIN_ROWS = 2
_EMPTY_CELL = "n/a"


# --------------------------------------------------------------------------
# Table repair
# --------------------------------------------------------------------------


def _split_row(line: str) -> list[str]:
    """Split a pipe-table row into cells, dropping the outer pipes."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|") and not stripped.endswith("\\|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in _PIPE.split(stripped)]


def _is_delimiter_row(line: str) -> bool:
    cells = _split_row(line)
    return bool(cells) and all(_DELIMITER_CELL.match(cell.replace(" ", "")) for cell in cells)


def _alignment(cell: str) -> str:
    cell = cell.replace(" ", "")
    left, right = cell.startswith(":"), cell.endswith(":")
    if left and right:
        return ":---:"
    if right:
        return "---:"
    if left:
        return ":---"
    return "---"


def _looks_numeric(cell: str) -> bool:
    return bool(re.match(r"^[`$(]*[-+]?\d[\d,.eE+\-%]*[`)%]*$", cell.strip()))


def _render_row(cells: Iterable[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _repair_table(block: list[str]) -> list[str]:
    """Normalise one contiguous pipe-table block to valid GFM geometry."""
    rows = [_split_row(line) for line in block]
    header = rows[0]

    if len(rows) >= _TABLE_MIN_ROWS and _is_delimiter_row(block[1]):
        delimiter_cells = rows[1]
        body = rows[2:]
    else:
        delimiter_cells = []
        body = rows[1:]

    # Widen to the widest row rather than truncating: a ragged row usually means
    # the model emitted a real extra value, and dropping data is worse than an
    # `n/a` in the header.
    width = max(len(header), *(len(row) for row in body)) if body else len(header)

    # Alignment: honour an existing delimiter row, otherwise right-align columns
    # whose first data value is numeric (the contract's `---:` rule).
    alignments = [_alignment(cell) for cell in delimiter_cells[:width]]
    while len(alignments) < width:
        index = len(alignments)
        numeric = any(len(row) > index and _looks_numeric(row[index]) for row in body)
        alignments.append("---:" if numeric else "---")

    def fit(cells: list[str]) -> list[str]:
        cells = [cell or _EMPTY_CELL for cell in cells[:width]]
        cells += [_EMPTY_CELL] * (width - len(cells))
        return cells

    out = [_render_row(fit(header)), _render_row(alignments)]
    out += [_render_row(fit(row)) for row in body]
    return out


# --------------------------------------------------------------------------
# Structural normalisation
# --------------------------------------------------------------------------


def _strip_outer_fence(lines: list[str]) -> list[str]:
    """Drop a code fence that wraps the entire document."""
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    last = next((i for i in range(len(lines) - 1, -1, -1) if lines[i].strip()), None)
    if first is None or last is None or first >= last:
        return lines
    open_match = _FENCE.match(lines[first])
    if not open_match:
        return lines
    info = open_match.group(2).strip().lower()
    if info not in ("", "markdown", "md", "gfm"):
        return lines
    close_match = _FENCE.match(lines[last])
    if not close_match or not lines[last].strip().startswith(open_match.group(1)[0]):
        return lines
    inner = lines[first + 1 : last]
    # Only unwrap when the fence really was the outermost container.
    if any(_FENCE.match(line) for line in inner) and not _fences_balanced(inner):
        return lines
    return inner


def _fences_balanced(lines: list[str]) -> bool:
    depth = 0
    marker = ""
    for line in lines:
        match = _FENCE.match(line)
        if not match:
            continue
        if depth == 0:
            depth, marker = 1, match.group(1)[0]
        elif match.group(1)[0] == marker:
            depth = 0
    return depth == 0


def _strip_preamble(lines: list[str]) -> list[str]:
    """Drop leading conversational lines ("Here is the report you asked for")."""
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if _PREAMBLE.match(stripped):
            index += 1
            continue
        break
    return lines[index:]


def _convert_setext(lines: list[str]) -> list[str]:
    out: list[str] = []
    in_code = False
    marker = ""
    for index, line in enumerate(lines):
        fence = _FENCE.match(line)
        if fence and (not in_code or fence.group(1)[0] == marker):
            in_code = not in_code
            marker = fence.group(1)[0] if in_code else ""
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        previous = out[-1] if out else ""
        is_text = bool(previous.strip()) and not _ATX.match(previous) and "|" not in previous
        if is_text and _SETEXT_H1.match(line):
            out[-1] = f"# {previous.strip()}"
            continue
        if is_text and _SETEXT_H2.match(line) and not _LIST_ITEM.match(previous):
            out[-1] = f"## {previous.strip()}"
            continue
        del index
        out.append(line)
    return out


def _normalise_headings(lines: list[str]) -> list[str]:  # noqa: C901
    """ATX-only headings, one H1, no skipped levels, no trailing hashes."""
    out: list[str] = []
    in_code = False
    marker = ""
    seen_h1 = False
    previous_level = 0
    for line in lines:
        fence = _FENCE.match(line)
        if fence and (not in_code or fence.group(1)[0] == marker):
            in_code = not in_code
            marker = fence.group(1)[0] if in_code else ""
            out.append(line)
            continue
        if in_code:
            out.append(line)
            continue
        match = _ATX.match(line)
        if not match:
            out.append(line)
            continue
        level = min(len(match.group(1)), _MAX_HEADING_LEVEL)
        text = match.group(2).strip()
        if not text:
            continue  # an empty heading is never meaningful
        if level == 1:
            if seen_h1:
                level = 2
            else:
                seen_h1 = True
        # Never skip a level (an H4 directly under an H2 becomes an H3).
        if previous_level and level > previous_level + 1:
            level = previous_level + 1
        previous_level = level
        out.append(f"{'#' * level} {text}")
    if not seen_h1:
        for index, line in enumerate(out):
            if _ATX.match(line):
                out[index] = f"# {_ATX.match(line).group(2).strip()}"  # type: ignore[union-attr]
                break
    return out


def _block_kind(line: str) -> str:  # noqa: PLR0911
    stripped = line.strip()
    if not stripped:
        return "blank"
    if _ATX.match(line):
        return "heading"
    if _FENCE.match(line):
        return "fence"
    if stripped.startswith(">"):
        return "quote"
    if _LIST_ITEM.match(line):
        return "list"
    if _PIPE.search(stripped):
        return "table"
    return "text"


def _rebuild_blocks(lines: list[str]) -> list[str]:
    """Repair tables and enforce one blank line between block-level elements."""
    out: list[str] = []
    in_code = False
    marker = ""
    table: list[str] = []

    def flush_table() -> None:
        if not table:
            return
        # Prose can legitimately contain a pipe; only treat the block as a table
        # when it has a delimiter row or every row is pipe-delimited.
        is_table = len(table) >= _TABLE_MIN_ROWS and (
            any(_is_delimiter_row(row) for row in table)
            or all(row.strip().startswith("|") for row in table)
        )
        repaired = _repair_table(table) if is_table else [row.rstrip() for row in table]
        _append_block(out, repaired, "table" if is_table else "text")
        table.clear()

    for line in lines:
        fence = _FENCE.match(line)
        if fence and (not in_code or fence.group(1)[0] == marker):
            flush_table()
            in_code = not in_code
            marker = fence.group(1)[0] if in_code else ""
            _append_block(out, [line.strip()], "fence")
            continue
        if in_code:
            out.append(line.rstrip())
            continue

        kind = _block_kind(line)
        if kind == "table":
            table.append(line)
            continue
        flush_table()
        if kind == "blank":
            if out and out[-1] != "":
                out.append("")
            continue
        _append_block(out, [line.rstrip()], kind)
    flush_table()
    return out


#: Block kinds that must be separated from their neighbours by a blank line.
_STANDALONE_BLOCKS = ("heading", "table", "fence", "quote")


def _append_block(out: list[str], block: list[str], kind: str) -> None:
    """Append a block, inserting the blank line the contract requires.

    Consecutive lines of the same kind stay glued together (a paragraph's lines,
    a list's items); everything else gets exactly one blank line between it and
    what came before.
    """
    if out and out[-1] != "":
        previous = _block_kind(out[-1])
        needs_gap = kind in _STANDALONE_BLOCKS or previous in _STANDALONE_BLOCKS or kind != previous
        if needs_gap:
            out.append("")
    out.extend(block)
    if kind in ("heading", "table"):
        out.append("")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------


def rewrite_figure_links(markdown: str, known_ids: Iterable[str]) -> tuple[str, list[str]]:
    """Drop image references to figures that do not exist in the manifest.

    Returns the cleaned markdown and the ids that were removed, so the caller
    can report them. Keeping a dangling ``figures/foo.png`` would render as a
    broken image in the UI and as a missing file in the downloaded bundle.
    """
    known = {str(fid) for fid in known_ids}
    removed: list[str] = []

    def replace(match: re.Match[str]) -> str:
        path_match = _FIGURE_PATH.match(match.group("path"))
        if path_match is None:
            return match.group(0)
        fid = path_match.group("fid")
        if fid in known:
            return f"![{match.group('alt')}](figures/{fid}.png)"
        removed.append(fid)
        return ""

    cleaned = _IMAGE.sub(replace, markdown)
    # An image that was alone on its line leaves an empty line behind.
    cleaned = re.sub(r"\n[ \t]*\n[ \t]*\n+", "\n\n", cleaned)
    return cleaned, sorted(set(removed))


def referenced_figures(markdown: str) -> list[str]:
    """Figure ids the document references, in first-appearance order."""
    seen: list[str] = []
    for match in _IMAGE.finditer(markdown):
        path_match = _FIGURE_PATH.match(match.group("path"))
        if path_match is None:
            continue
        fid = path_match.group("fid")
        if fid not in seen:
            seen.append(fid)
    return seen


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def normalize_markdown(markdown: str) -> str:
    """Return ``markdown`` repaired to the report markdown contract."""
    if not markdown:
        return ""
    text = markdown.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n")
    # Non-breaking space and the Unicode line/paragraph separators break both
    # renderers and diffs, and LLM output carries them often enough to fold here.
    text = text.replace("\u00a0", " ").replace("\u2028", "\n").replace("\u2029", "\n")
    lines = [_TRAILING_WS.sub("", line.expandtabs(4)) for line in text.split("\n")]
    # Chatter first, then the fence: models often say "Here is the report" and
    # *then* wrap the document, so the fence is only outermost once that is gone.
    lines = _strip_preamble(lines)
    lines = _strip_outer_fence(lines)
    lines = _strip_preamble(lines)
    lines = [_BULLET.sub(r"\1- ", line) for line in lines]
    lines = _convert_setext(lines)
    lines = _normalise_headings(lines)
    lines = _rebuild_blocks(lines)

    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def lint_markdown(  # noqa: C901, PLR0912
    markdown: str, known_figure_ids: Iterable[str] = ()
) -> list[str]:
    """Report contract violations that survive normalisation."""
    issues: list[str] = []
    lines = markdown.split("\n")
    known = {str(fid) for fid in known_figure_ids}

    in_code = False
    marker = ""
    h1_count = 0
    heading_count = 0
    previous_level = 0
    table_widths: list[int] = []

    for number, line in enumerate(lines, start=1):
        fence = _FENCE.match(line)
        if fence and (not in_code or fence.group(1)[0] == marker):
            in_code = not in_code
            marker = fence.group(1)[0] if in_code else ""
            continue
        if in_code:
            continue

        match = _ATX.match(line)
        if match:
            level = len(match.group(1))
            heading_count += 1
            if level == 1:
                h1_count += 1
            if previous_level and level > previous_level + 1:
                issues.append(
                    f"line {number}: heading level jumps from H{previous_level} to H{level}"
                )
            previous_level = level
            continue

        if "|" in line and _PIPE.search(line.strip()):
            width = len(_split_row(line))
            table_widths.append(width)
        elif table_widths:
            if len(table_widths) < _TABLE_MIN_ROWS:
                pass  # prose containing a pipe
            elif len(set(table_widths)) > 1:
                issues.append(
                    f"line {number - 1}: table has ragged rows ({sorted(set(table_widths))} cells)"
                )
            table_widths = []

        if "\t" in line:
            issues.append(f"line {number}: literal tab character")
        if re.search(r"<[a-zA-Z/][^>]*>", line):
            issues.append(f"line {number}: raw HTML is not allowed")

    if in_code:
        issues.append("unclosed fenced code block")
    if heading_count and h1_count == 0:
        issues.append("document has no level-1 title")
    if h1_count > 1:
        issues.append(f"document has {h1_count} level-1 headings (expected exactly 1)")
    if re.search(r"\n[ \t]*\n[ \t]*\n", markdown):
        issues.append("consecutive blank lines")

    if known:
        issues.extend(
            f"references unknown figure id '{fid}'"
            for fid in referenced_figures(markdown)
            if fid not in known
        )
    return issues
