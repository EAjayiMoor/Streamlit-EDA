# AGENTS.md

## Project direction

This repository is evolving from a healthcare elective-care analytics dashboard into a general-purpose exploratory data analysis (EDA) application. Healthcare analytics is the first domain implementation, not the product boundary.

The product should help a user bring in a dataset, understand its structure and quality, explore distributions and relationships, create useful visualisations, and export or share findings without writing code.

## Working principles

- Keep the core EDA workflow domain-agnostic.
- Treat healthcare-specific loaders, metrics, labels, and pages as a domain adapter or example workspace.
- Treat PAH as a configurable pilot organisation, never as an implicit application-wide constant.
- Design ingestion for batches of historical NHS England CSV extracts, including multiple providers and reporting periods.
- Prefer configuration and metadata over hard-coded column names, page names, or business rules.
- Preserve a clear separation between ingestion, profiling, transformation, analysis, visualisation, and presentation.
- Keep raw data immutable. Write derived data and exports to explicit output locations.
- Never commit secrets, API keys, local environment files, generated caches, or private source data by accident.
- Do not log, display, or include sensitive values in diagnostics.
- Make missing values, inferred types, filters, and transformations visible to the user.
- Preserve source file, organisation, reporting period, and data vintage metadata through every transformation.
- Never combine hospitals or periods silently; expose the selected scope and aggregation level in every view.
- Add tests for reusable logic before changing domain-specific behaviour.

## Moorhouse UI standard

The application must align with the supplied Moorhouse UI Standard
(Moorhouse-UI-Standard.md). Treat that document as the source of truth for
visual design and interaction decisions.

- Use the Moorhouse tokens, Poppins typography, spacing scale, radii, shadows,
  and chart palette; do not invent a parallel visual language.
- Follow the principles “credible, not flashy” and “data first”: no gradients,
  glass effects, decorative illustrations, coloured shadows, or unnecessary
  animation in working views.
- Use British English, sentence case, plainspoken microcopy, and verb-first
  action labels.
- Design to WCAG 2.1 AA, with visible keyboard focus, labelled controls, and
  status conveyed by text or icons as well as colour.
- Apply the standard to Streamlit equivalents of the specified components and
  document any unavoidable framework limitation.
- Keep the page structure consistent: workflow selection, upload and validation,
  profile, analysis, comparison, and export should be recognisable across modules.
- Build pages from the shared application shell and reusable layout primitives;
  do not create a bespoke navigation model for each healthcare workflow.

## Target architecture

Use these conceptual layers as the application is refactored:

1. src/ingestion/ — file and connector inputs with a common dataset contract.
2. src/profiling/ — schema, quality, summary, and anomaly profiling.
3. src/transforms/ — reusable, user-visible transformations.
4. src/analysis/ — generic descriptive statistics and relationship analysis.
5. src/visualisation/ — chart specifications and rendering helpers.
6. src/domains/ — optional domain adapters, including the current healthcare implementation.
7. pages/ — thin Streamlit presentation and interaction code.

Do not perform a large-bang rewrite. Introduce generic interfaces around the current working code and migrate one workflow at a time.

## Delivery expectations

- Update docs/PRODUCT.md when product scope or priorities change.
- Update docs/SPRINT_PLAN.md when sprint scope or acceptance criteria change.
- Run focused tests and a Streamlit smoke check where practical.
- Record known limitations rather than silently changing outputs.
- Keep commits small and describe the user-visible outcome.
