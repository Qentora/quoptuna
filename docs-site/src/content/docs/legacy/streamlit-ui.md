---
title: Legacy Streamlit UI
description: The original Streamlit dashboard, kept as a fallback to the supported Next.js wizard.
---

:::caution
This is a **legacy** interface. The supported UI is the Next.js wizard. Use the Streamlit dashboard only as a fallback.
:::

QuOptuna ships an older multi-page **Streamlit** dashboard (`src/quoptuna/frontend/app.py`). It predates the Next.js UI and is retained for compatibility, but it is not the recommended way to use QuOptuna.

## What it is

The Streamlit dashboard follows the same conceptual flow as the modern wizard — **dataset → optimize → train → SHAP analysis → report** — through a set of Streamlit pages rather than the guided 6-step wizard. It talks to the same optimization engine underneath.

## How to launch it

```bash
quoptuna run --streamlit
```

Or via the Make target:

```bash
make run_streamlit
```

## Should you use it?

Prefer the Next.js wizard for any new work — it is the actively supported UI and shares the same underlying workflow engine. Reach for the Streamlit dashboard only if the Next.js UI is unavailable in your environment.

## See also

- [Architecture](/explanation/architecture/)
- [Feature overview](/explanation/features/)
