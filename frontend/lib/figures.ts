/**
 * Figure identity shared with the backend report bundle.
 *
 * A generated report references its plots as `figures/<figure_id>.png`. Those
 * paths resolve to real files inside the downloaded research dump, and to the
 * in-memory data URLs below when the report is rendered in the browser — so the
 * same markdown works in both places.
 *
 * The id rule mirrors `report_context.figure_id_for_plot` in
 * `src/quoptuna/backend/xai/report_context.py`; keep the two in step.
 */

import type { FairnessResult } from '@/components/optimizer/types';

// Plot keys produced by the SHAP explainer; everything else in `analysis.plots`
// is a performance curve.
const SHAP_PLOT_KEYS = new Set([
  'bar',
  'beeswarm',
  'violin',
  'heatmap',
  'waterfall',
  'force',
  'decision',
]);

/** camelCase / PascalCase -> snake_case, matching the backend's `_snake`. */
function snake(name: string): string {
  return name.replace(/(?!^)([A-Z])/g, '_$1').toLowerCase();
}

/** Canonical figure id for an `analysis.plots` key (`bar` -> `shap_bar`). */
export function figureIdForPlot(key: string): string {
  const id = snake(key);
  return SHAP_PLOT_KEYS.has(id) ? `shap_${id}` : id;
}

export interface FigureSources {
  plots?: Record<string, string> | null;
  confusionMatrixPlot?: string | null;
  fairness?: FairnessResult | null;
}

/**
 * Map every available figure id to its image data URL.
 *
 * Used to resolve `figures/<id>.png` references while rendering a report, so the
 * plots the agent wrote about appear inline instead of as broken images.
 */
export function buildFigureMap(sources: FigureSources): Record<string, string> {
  const map: Record<string, string> = {};
  if (sources.confusionMatrixPlot) {
    map.confusion_matrix = sources.confusionMatrixPlot;
  }
  for (const [key, value] of Object.entries(sources.plots ?? {})) {
    if (value) map[figureIdForPlot(key)] = value;
  }
  const fairness = sources.fairness;
  if (fairness) {
    for (const [key, value] of Object.entries(fairness.plots ?? {})) {
      if (value) map[`fairness_${snake(key)}`] = value;
    }
    if (fairness.mitigation?.comparison_plot) {
      map.fairness_mitigation_comparison = fairness.mitigation.comparison_plot;
    }
  }
  return map;
}

/** Resolve a markdown image path to a data URL, or null when it is not a figure. */
export function resolveFigureSrc(
  src: string | undefined,
  figures: Record<string, string>
): string | null {
  if (!src) return null;
  const match = /^(?:\.\/)?figures\/([A-Za-z0-9_.-]+)\.(?:png|jpg|jpeg|svg)$/.exec(src);
  if (!match) return null;
  return figures[match[1]] ?? null;
}
