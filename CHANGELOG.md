# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [0.1.3]
### Changed
- Migrated the documentation from MkDocs to an Astro + Starlight site in `docs-site/`,
  reorganized on the Diátaxis framework (tutorials / how-to / reference / explanation)
  with a custom marketing landing page.
- Rewrote docs against the current Next.js + FastAPI app; added architecture, feature,
  CLI, REST API, configuration, and model-catalog pages; consolidated the legacy
  Streamlit docs into a single page.
- Replaced the MkDocs GitHub Actions workflows with an Astro Pages deploy plus a
  per-PR docs preview build; removed MkDocs dependencies from `pyproject.toml`.

### Fixed
- Corrected the fabricated `create_study` example (README and docs) to the real
  `Optimizer` API, and the MIT/Apache-2.0 license inconsistency (Apache 2.0).
