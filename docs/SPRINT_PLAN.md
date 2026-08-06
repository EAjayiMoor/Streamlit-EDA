# Sprint plan: Generalised EDA Workbench

Assumption: six two-week sprints, with a review and scope reset at the end of each sprint. The plan prioritises a usable generic path while protecting the current healthcare use case.

## Sprint 0 — Baseline and product contract

Goal: establish a safe, measurable starting point.

Deliverables:

- Capture the current application flow and key healthcare pages.
- Add a minimal test and smoke-check approach.
- Define the dataset, schema, profile, filter, and finding contracts.
- Identify sensitive-data handling rules and supported file limits.
- Confirm a representative fixture dataset that contains no private healthcare data.
- Translate the Moorhouse UI Standard into a Streamlit implementation checklist.
- Define the initial theme tokens, typography, spacing, chart palette, and accessibility baseline.
- Define the canonical healthcare dimensions: organisation, provider, dataset family, reporting period, and source vintage.
- Define the landing-page workflow cards and shared upload journey for RTT, referrals, theatre, outpatient, inpatient, workforce, and finance.
- Agree the shared page structure: landing, upload, validation, profile, analysis, comparison, and findings/export.

Acceptance criteria:

- A new contributor can run the app and understand the current entry points.
- The generic contracts are documented.
- Baseline behaviour for the current healthcare workspace is recorded.
- A design baseline is agreed against the Moorhouse UI Standard, including known Streamlit gaps.
- PAH is represented as configuration and fixture data, not as a required hard-coded field.
- The shared page structure and navigation shell are approved before workflow-specific page work begins.

## Sprint 1 — Landing page and batch CSV ingestion

Goal: let a user choose an analysis flow from the landing page and load one or more CSV files.

Deliverables:

- CSV upload flow for the initial healthcare workflows.
- Multi-file CSV batch upload with progress and per-file results.
- Landing-page workflow cards for RTT, referrals, theatre, outpatient, inpatient, workforce, and finance.
- Shared application shell with header, breadcrumb, workflow context, scope bar, and consistent navigation.
- Dataset workspace state and dataset summary.
- Common ingestion result with errors and warnings.
- File-size and malformed-file handling.
- Sample dataset for development and demonstration.
- Source manifest recording file, dataset family, organisation, period, and vintage.
- Shared Moorhouse theme foundation for page background, surfaces, typography, controls, and status states.

Acceptance criteria:

- A user can load a supported CSV or Excel file and see rows, columns, and load warnings.
- An invalid file produces an actionable message rather than a traceback.
- No API key or external service is required for the core workflow.
- The load flow uses Moorhouse tokens, sentence-case microcopy, visible focus, and accessible control labels.
- A batch can contain multiple historical periods and providers without losing file provenance.
- Incompatible files are reported individually and are not silently appended.
- The landing page can route a user into the selected workflow with the correct upload and validation configuration.
- The same shell and page sequence is used regardless of which workflow card is selected.

## Sprint 2 — Canonical schema and data quality

Goal: answer “what is in this dataset?” reliably and make compatible NHS extracts comparable.

Deliverables:

- Type inference with manual override.
- Column profile cards for numeric, categorical, text, boolean, and date fields.
- Missingness, uniqueness, duplicates, and constant-column checks.
- Quality summary with links to affected columns.
- Dataset-family schema mappings and column alias handling.
- Normalisation of organisation names, reporting periods, dates, measures, and categories.
- Duplicate-period and missing-period checks.

Acceptance criteria:

- Profiling works on numeric-only, mixed-type, and time-indexed fixtures.
- Inferred types can be corrected without modifying raw data.
- Profile results are covered by focused tests.
- A test batch with changed column names and multiple reporting periods produces a visible validation report.
- Raw source files remain traceable after standardisation.

## Sprint 3 — Generic and provider-aware exploration

Goal: provide useful analysis without hard-coded business fields.

Deliverables:

- Generic dimension and measure selectors.
- Univariate, bivariate, categorical, and time-series charts.
- Active filter state and filtered-row counts.
- Empty-state and incompatible-column handling.
- House chart styling: canonical series colours, horizontal gridlines, accessible legends, and exact-value tooltips.
- Scope controls for organisation, provider group, dataset family, period, and source vintage.
- Reusable healthcare view configuration for RTT, flow, referrals, outpatient activity, theatre, and financial analysis.
- Workflow-specific expected columns, quality checks, and metric definitions.
- Generic profile page and reusable empty, loading, warning, and error states.

Acceptance criteria:

- A user can create at least four chart types from a new dataset.
- Charts update when filters or selected fields change.
- Healthcare-specific terminology is absent from the generic pages.
- Generic pages conform to the Moorhouse layout, typography, spacing, component, and chart rules.
- Every healthcare view makes organisation, period, denominator, and source scope visible.
- Each workflow exposes its analysis through the agreed page structure rather than a bespoke page hierarchy.

## Sprint 4 — Findings, comparisons, and exports

Goal: turn exploration into shareable work.

Deliverables:

- Saved finding containing dataset, fields, filters, and chart configuration.
- Export of profile summary, filtered data, and chart image or HTML where supported.
- Basic workspace reset and repeatability behaviour.
- Plain-language observation templates grounded in displayed statistics.
- Peer and historical comparison configuration with compatible-scope checks.
- Exported findings include provider, period, dataset family, vintage, and denominator metadata.
- Findings and exports follow the shared page structure and use the Moorhouse component and microcopy rules.

Acceptance criteria:

- Another user can understand how an exported finding was produced.
- Exported data respects active filters.
- No generated output contains secrets or unnecessary raw sensitive values.
- Comparisons are blocked or clearly qualified when definitions or periods are not comparable.

## Sprint 5 — Healthcare adapter and PAH generalisation

Goal: prove the generalized foundation can support the existing RTT use case for any compatible hospital.

Deliverables:

- Healthcare workspace boundary and adapter configuration.
- Migrate one high-value workflow first, such as RTT backlog or executive summary.
- Preserve existing loaders and specialised metrics behind the adapter.
- Regression checks for migrated views.
- Replace PAH-specific assumptions with organisation configuration.
- Run the migrated workflow against PAH plus at least one synthetic peer organisation.
- Defer forecasting and predictive modelling; focus on descriptive EDA, backlog, flow, quality, and comparison views.

Acceptance criteria:

- The generic EDA workflow remains usable with a non-healthcare fixture.
- The selected healthcare workflow produces equivalent or explained-difference results.
- Domain-specific code is isolated from generic profiling and exploration services.
- Switching organisation changes the results through data and configuration, not duplicated page code.

## Sprint 6 — NHS batch pilot and beta hardening

Goal: make the workbench dependable for pilot users.

Deliverables:

- Performance checks on realistic file sizes.
- Pilot import of a representative NHS England historical CSV batch.
- Reconciliation report showing loaded, rejected, duplicate, and missing-period files.
- Accessibility and responsive-layout pass.
- Visual QA against the Moorhouse UI Standard at desktop, mobile, keyboard-only, and 200% zoom.
- Error logging that excludes secrets and sensitive row values.
- User documentation and a short guided demo.
- Beta feedback review and next-roadmap proposal.

Acceptance criteria:

- Core flows have smoke coverage.
- Known limitations and supported formats are documented.
- A pilot user can complete load, profile, explore, and export without developer support.
- The beta meets the Moorhouse UI and WCAG 2.1 AA acceptance checklist, with deviations recorded.
- A pilot user can compare hospitals across compatible historical periods and trace every result to its source files.

## Prioritisation rules

When trade-offs are required:

1. Protect data safety and prevent secret leakage.
2. Keep the generic workflow domain-agnostic.
3. Preserve a working healthcare path.
4. Prefer explainable, testable features over automation that makes unsupported claims.
5. Defer connectors, collaboration, and ML until the file-based MVP is stable.

## Immediate next actions

- Create a small, synthetic fixture dataset and baseline tests.
- Implement a generic dataset state object with batch manifest and provenance.
- Add canonical organisation and reporting-period dimensions.
- Use RTT as the first supported NHS England batch format.
- Keep forecasting out of the initial RTT scope.
- Build the RTT workflow through the shared landing, upload, validation, profile, analysis, comparison, and findings pages.
- Move profiling logic out of healthcare-specific pages.
- Add a generic “Data profile” page.
- Decide whether the current AI chat page is retained as an optional provider integration or removed from the MVP.
