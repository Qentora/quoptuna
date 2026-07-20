# Reusable assets (reference, don't duplicate)

Point posts at these existing files in the repo instead of copying binaries into `blogs/`. Paths are relative to the repo root.

## Branding
| Use | Path | Where |
|---|---|---|
| GitHub social preview / OG image / post hero | `assets/branding/social-preview.png` | HN URL card, PH gallery, X tweet 1, dev.to cover |
| Full logo (light) | `assets/branding/logo-full.png` | README header, LinkedIn banner |
| Logo mark (square/icon) | `assets/branding/logo-mark.png` | avatars, favicons, small placements |

**Raw GitHub URL pattern** (for embedding in dev.to/Hashnode/Medium — replace `main` if needed):
`https://raw.githubusercontent.com/Qentora/quoptuna/main/assets/branding/social-preview.png`

## SHAP / results visuals (real output — great for blog embeds)
Located in `experiments/basic_dataset_test/blood/`:

| File | Best for |
|---|---|
| `beeswarm.png` | The hero explainability image — most legible SHAP plot |
| `waterfall.png` | Single-prediction explanation (good for "explain the winner" sections) |
| `bar.png` | Feature-importance summary |
| `violin.png` | Distribution of SHAP values |
| `heatmap.png` | Instance-level overview |

Use `beeswarm.png` in the dev.to/Medium articles and the X thread (tweet 5), `waterfall.png` where you show a single explained prediction.

## Still needed (flagged in main README TODOs)
- ❌ **Demo GIF / 60-sec video** of the 6-step wizard — record and add; embed everywhere.
- ❌ **Screenshot of the web wizard** — for PH gallery + README.
- ❌ **Hosted live demo link** — biggest conversion lever; deploy read-only (HF Spaces / Railway).

> Do not commit new large binaries into `blogs/` — reference the repo paths above so there's a single source of truth.
