"""Regression tests for the report markdown contract enforcement.

The agents are prompted to produce strict GFM, but prompting is not a guarantee -
these cover the repairs that make a generated report safe to commit to a paper.
"""

from quoptuna.backend.xai.markdown_report import (
    lint_markdown,
    normalize_markdown,
    referenced_figures,
    rewrite_figure_links,
)


def test_unwraps_outer_fence_and_chatty_preamble():
    raw = "Here is the report you requested:\n\n```markdown\n# Title\n\nBody.\n```\n"
    assert normalize_markdown(raw) == "# Title\n\nBody.\n"


def test_converts_setext_and_unicode_bullets():
    raw = "Title\n=====\nSection\n-------\n• first\n• second\n"
    out = normalize_markdown(raw)
    assert out.startswith("# Title\n")
    assert "## Section" in out
    assert "- first" in out
    assert "•" not in out


def test_only_one_h1_survives_and_levels_never_skip():
    raw = "# First\n\n#### Deep\n\n# Second\n"
    out = normalize_markdown(raw)
    assert out.count("\n# ") + out.startswith("# ") == 1
    assert "## Deep" in out  # H4 under H1 is demoted to H2, not left skipping
    assert "## Second" in out


def test_repairs_table_geometry_without_losing_cells():
    raw = "| Metric | Value |\n| F1 | 0.9 | 12 |\n| Accuracy |  |\n"
    out = normalize_markdown(raw)
    rows = [line for line in out.splitlines() if line.startswith("|")]
    assert rows[1].startswith("| ---")  # a delimiter row was inserted
    widths = {row.count("|") for row in rows}
    assert len(widths) == 1  # every row has the same cell count
    assert "12" in out  # the extra value was kept, not truncated
    assert "n/a" in out  # the empty cell was filled


def test_blank_lines_are_normalised_around_blocks():
    raw = "# Title\nParagraph.\n## Section\n\n\n\nMore.\n"
    out = normalize_markdown(raw)
    assert out == "# Title\n\nParagraph.\n\n## Section\n\nMore.\n"
    assert not lint_markdown(out)


def test_pipes_inside_headers_do_not_ragged_the_table():
    raw = "| Feature | Mean \\|SHAP\\| |\n| --- | ---: |\n| age | 0.41 |\n"
    assert not lint_markdown(normalize_markdown(raw))


def test_invented_figure_references_are_dropped():
    raw = "# T\n\n![a](figures/shap_bar.png)\n\n![b](figures/imaginary.png)\n"
    cleaned, removed = rewrite_figure_links(raw, ["shap_bar"])
    assert removed == ["imaginary"]
    assert referenced_figures(cleaned) == ["shap_bar"]
    assert not lint_markdown(normalize_markdown(cleaned), ["shap_bar"])


def test_lint_reports_what_it_cannot_repair():
    issues = lint_markdown("## No title\n\n<div>raw html</div>\n")
    assert any("no level-1 title" in issue for issue in issues)
    assert any("raw HTML" in issue for issue in issues)


def test_empty_input_is_not_an_error():
    assert normalize_markdown("") == ""
    assert lint_markdown("") == []
