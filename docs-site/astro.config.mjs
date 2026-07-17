// @ts-check
import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

// Deployed to GitHub Pages project site: https://Qentora.github.io/quoptuna
// so we need an explicit `site` + `base`. Override at build time for PR previews
// with DOCS_SITE / DOCS_BASE env vars (see .github/workflows/docs-preview.yml).
const site = process.env.DOCS_SITE ?? "https://Qentora.github.io";
const base = process.env.DOCS_BASE ?? "/quoptuna";

export default defineConfig({
  site,
  base,
  trailingSlash: "ignore",
  integrations: [
    starlight({
      title: "QuOptuna",
      description:
        "Tune quantum and classical machine-learning models in a single, " +
        "fairness-aware, explainable hyperparameter search. Built on Optuna and PennyLane.",
      logo: {
        src: "./src/assets/logo.png",
        alt: "QuOptuna",
      },
      favicon: "/favicon.png",
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/Qentora/quoptuna",
        },
      ],
      editLink: {
        baseUrl: "https://github.com/Qentora/quoptuna/edit/main/docs-site/",
      },
      customCss: ["./src/styles/theme.css"],
      lastUpdated: true,
      // Custom marketing landing page uses the splash template; components below
      // let the homepage render bespoke sections while docs pages keep the shell.
      components: {},
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { slug: "getting-started/installation" },
            { slug: "getting-started/quickstart" },
          ],
        },
        {
          label: "Tutorials",
          items: [
            { slug: "tutorials/first-optimization-ui" },
            { slug: "tutorials/optimize-from-cli" },
            { slug: "tutorials/python-api" },
          ],
        },
        {
          label: "How-to Guides",
          items: [
            { slug: "how-to/choose-samplers-and-pruners" },
            { slug: "how-to/tune-for-speed-and-quality" },
            { slug: "how-to/run-fairness-aware-search" },
            { slug: "how-to/use-multiclass" },
            { slug: "how-to/generate-reports" },
            { slug: "how-to/deploy-this-site" },
          ],
        },
        {
          label: "Reference",
          items: [
            { slug: "reference/cli" },
            { slug: "reference/rest-api" },
            { slug: "reference/python-api" },
            { slug: "reference/models" },
            { slug: "reference/configuration" },
          ],
        },
        {
          label: "Explanation",
          items: [
            { slug: "explanation/architecture" },
            { slug: "explanation/optimization-engine" },
            { slug: "explanation/workflow-engine" },
            { slug: "explanation/features" },
          ],
        },
        {
          label: "More",
          items: [
            { slug: "legacy/streamlit-ui" },
            { slug: "contributing" },
          ],
        },
      ],
    }),
  ],
});
