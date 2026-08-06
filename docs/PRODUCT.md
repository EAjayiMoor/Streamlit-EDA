# Product brief: Generalised EDA Workbench

## Product summary

The project will become a configurable, self-service EDA workbench for analysts, operators, and decision-makers. A user should be able to upload or select a dataset, understand what is in it, assess its quality, explore it visually, and leave with reproducible findings.

The existing healthcare elective-care dashboard becomes the first domain workspace
built on the shared EDA foundation. Its specialist RTT, theatre, referral,
workforce, and financial analyses should remain available while no longer
defining the navigation or data model for the whole application. PAH becomes
the first configured organisation, not a hard-coded product assumption.

## Problem

Many users have data but cannot quickly answer basic questions about its shape, quality, patterns, outliers, and relationships. Existing tools often require technical knowledge, produce disconnected charts, or encode assumptions that only make sense for one sector.

## Target users

- Analysts who need a fast first-pass investigation before deeper modelling.
- Operational or business users who need trustworthy summaries without writing Python.
- Data teams who want a repeatable profiling and exploration starting point.
- Domain specialists who need to add sector-specific metrics on top of generic EDA.

## Product promise

Given a supported dataset, the workbench will make the path from “what is this data?” to “what should I investigate next?” clear, inspectable, and repeatable.

## MVP capabilities

1. Dataset workspace
   - Upload CSV and Excel files.
   - Show dataset name, size, rows, columns, and load status.
   - Allow a user to switch between datasets in a session.

2. Automatic profiling
   - Infer column types with an override option.
   - Show missingness, uniqueness, cardinality, and basic distributions.
   - Flag duplicate rows, constant columns, suspicious date/numeric fields, and high-missingness columns.

3. Exploration
   - Select dimensions and measures without hard-coded domain names.
   - Generate common univariate, bivariate, time-series, and categorical views.
   - Filter data and show the active filter state.
   - Display plain-language observations with the underlying values visible.

4. Reproducibility and export
   - Show the steps or configuration used to create a view.
   - Export filtered data, profile summaries, and selected charts.
   - Keep transformations explicit and reversible within a workspace.

5. Domain adapters
   - Preserve the existing healthcare analyses behind a healthcare workspace.
   - Define a documented adapter pattern for future domains such as finance, operations, or sales.

6. Batch historical NHS data
   - Upload a batch of NHS England CSV extracts in one operation.
   - Register source file, dataset family, provider or hospital, reporting period, and vintage metadata.
   - Append compatible files into a canonical analytical dataset without losing provenance.
   - Detect incompatible schemas, duplicate periods, missing periods, and provider-name variations before loading.
   - Filter every view by organisation, dataset family, reporting period, and source vintage.

7. Multi-hospital healthcare views
   - Compare one hospital with peers, a selected provider group, or a national/region-level benchmark when the source supports it.
   - Retain the current PAH views as reusable templates driven by organisation and metric configuration.
   - Make rankings, benchmarks, and denominators visible so comparisons are not misinterpreted.

The first NHS England dataset family for the generalized healthcare workflow
will be RTT data. This provides the initial end-to-end path for batch ingestion,
historical analysis, provider comparison, data-quality checks, and backlog
views. Forecasting is deferred until the core ingestion and EDA workflows are
stable. Outpatient, referral, theatre, inpatient, workforce, and financial
datasets follow as subsequent adapters.

## Landing-page workflow

The landing page is the starting point for the healthcare EDA experience. It
should present clear workflow cards or tiles:

- RTT
- Referrals
- Theatre
- Outpatient activity
- Inpatient activity
- Workforce
- Finance

Each workflow leads to the same guided journey:

1. Choose the workflow.
2. Upload one or more relevant CSV files.
3. Review detected files, periods, organisations, columns, and validation warnings.
4. Confirm the data scope and continue.
5. Open the workflow-specific EDA views.
6. Filter, compare, and export findings.

The upload and validation experience should be shared across workflows. The
analysis views, metric definitions, expected columns, and quality rules may be
specific to each workflow adapter.

## Out of scope for MVP

- Automated causal claims or decision-making.
- Production-grade multi-user collaboration and permissions.
- Arbitrary database connectors before the file-based workflow is stable.
- A full no-code machine-learning platform.
- Removing healthcare functionality before the generic foundation can support it.

## Product principles

- Generic by default; specialised by configuration or adapter.
- Explainable before clever.
- Safe handling of sensitive data.
- A useful result in minutes.
- Every insight should be traceable to data, filters, and a visible method.

## Success measures

- A new user can load a CSV and reach a useful profile in under five minutes.
- The core profiling and exploration workflow contains no healthcare-specific field names.
- At least three dataset shapes are supported: purely numeric, mixed tabular, and time-indexed data.
- A healthcare dataset can still produce the current key operational views through its adapter.
- A batch of historical NHS England CSVs can be loaded, validated, and filtered by hospital and reporting period.
- PAH-specific views work when PAH is selected as the organisation, without PAH-specific code in generic pages.
- A user can start from the landing page, select a healthcare workflow, upload CSV data, and reach relevant EDA views without developer intervention.
- A user can export a finding with enough context for another person to reproduce it.

## Moorhouse design alignment

The product UI will align with the Moorhouse UI Standard
(Moorhouse-UI-Standard.md). This is a product requirement and a shared
foundation for the generalized EDA workbench, not a final styling pass.

The implementation will use the standard's:

- Deep purple, teal, ocean, orange, semantic, and status colour tokens.
- Poppins UI typography, the defined type scale, and tabular numerals for data.
- 4px spacing scale, specified radii, borders, and limited elevation.
- Credible, data-first layouts with calm default states and colour reserved for meaning.
- House chart conventions, including series order, horizontal gridlines, and accessible legends.
- Sentence-case British English microcopy and verb-first action labels.
- WCAG 2.1 AA requirements, keyboard parity, visible focus, responsive reflow, and reduced motion.

Streamlit-specific components should reproduce the intent and states of the
standard's buttons, fields, cards, navigation, tabs, data tables, empty states,
toasts, and chart treatments. Any deviation must be documented and justified.

## Proposed navigation

- Home / Workspaces
- Choose analysis workflow
- Upload CSV data
- Data profile
- Explore
- Compare
- Findings and exports
- Optional domain workspace (Healthcare)

## Page structure

All healthcare workflows should use the same page journey and visual hierarchy.
Only the data contract, quality rules, metrics, and analysis views should change
between RTT, referrals, theatre, outpatient, inpatient, workforce, and finance.

### 1. Landing page — choose a workflow

Purpose: orient the user and start a piece of analysis.

- Moorhouse product header and concise product description.
- Workflow cards for RTT, referrals, theatre, outpatient, inpatient, workforce, and finance.
- Each card states the expected CSV data and the analysis available.
- Recent or active workspaces may appear below the primary workflow choices.
- One clear primary action per card; no dashboard metrics before a dataset is selected.

### 2. Upload page — add data

Purpose: collect one or more CSV files for the selected workflow.

- Breadcrumb showing the selected workflow.
- Upload zone with supported file type, size, and privacy guidance.
- Batch file list with filename, detected dataset family, period, organisation, and status.
- Clear action to validate the files.
- Empty, loading, success, warning, and error states.

### 3. Validation page — confirm scope

Purpose: let the user understand and approve what will be analysed.

- File-level validation results and actionable corrections.
- Detected organisations, reporting periods, row counts, columns, and missingness.
- Duplicate, incompatible, or missing-period warnings.
- Scope controls for organisation, period, provider group, and source vintage.
- Primary action to open the analysis only when the scope is understood.

### 4. Profile page — understand the data

Purpose: provide a generic EDA summary before domain interpretation.

- Dataset summary and provenance.
- Data-quality scorecard with visible definitions.
- Column-level profile table and distributions.
- Missingness, duplicates, type warnings, and unusual values.
- Consistent export of the profile report.

### 5. Analysis page — answer workflow questions

Purpose: present the selected workflow's core views.

- Workflow title, selected organisation, period, and source scope above the content.
- A compact KPI or status strip for the most important measures.
- Two- or three-column working layout for charts, tables, and filters.
- Tabs or secondary navigation for related views, with no more than eight top-level items.
- Explanatory text beside the evidence, not unsupported automated conclusions.
- RTT examples: backlog, waiting-time bands, demand and throughput, flow, and pathway mix.

### 6. Compare page — examine change and peers

Purpose: compare hospitals, providers, periods, or segments where definitions are compatible.

- Comparison scope and denominator displayed prominently.
- Side-by-side or trend views using the Moorhouse chart rules.
- Warnings when periods, definitions, or denominators are not comparable.
- Clear distinction between actual values, benchmarks, and missing data.

### 7. Findings page — save and export

Purpose: make analysis reusable and shareable.

- Saved finding title, short observation, evidence, filters, and provenance.
- Export actions for filtered data, profile summaries, charts, and finding metadata.
- Plainspoken completion and error states.
- No raw data or sensitive values included unless explicitly selected by the user.

### Shared shell rules

- Use one consistent header, workflow breadcrumb, page title, scope bar, and navigation pattern.
- Use the Moorhouse 12-column layout, responsive collapse below the medium breakpoint, and standard spacing.
- Keep working views dense and decision views more generous.
- Use the Moorhouse palette and chart series order; colour is never the only status signal.
- Keep organisation, reporting period, dataset family, and source vintage visible in the scope bar.
- Every page must have a useful empty state and a clear next action.

## Architecture direction

The Streamlit pages should become thin views over reusable services. A dataset
contract should carry the table, schema metadata, quality results, active filters,
and transformation history. A batch ingestion contract should additionally carry
source file, dataset family, provider, period, and vintage metadata.

Domain modules may add metrics and terminology, but generic pages must not import
healthcare loaders directly. The healthcare adapter should expose a canonical
provider-period model so that the current PAH views can operate over any
compatible hospital.

The current src/data, src/transforms, src/metrics, src/models, and pages code should be migrated incrementally. The first refactor should establish generic profiling and dataset state without attempting to rewrite every existing healthcare page.

## Risks and mitigations

- Sensitive data exposure: use local-first processing, explicit upload messaging, redacted diagnostics, and strict ignore rules.
- Scope expansion: treat connectors, collaboration, and ML as later roadmap items.
- Premature modelling: keep forecasting and predictive features out of the initial workflow until descriptive EDA is proven.
- Regression in healthcare views: retain the current domain workspace and add smoke tests around key outputs.
- NHS extract variation: maintain per-dataset-family schema mappings, validation reports, and explicit versioning.
- Misleading comparisons: require compatible periods, definitions, denominators, and visible peer scope before showing benchmarks.
- Ambiguous automated insights: show evidence and confidence limits; avoid unsupported explanations.
