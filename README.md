# Moorhouse healthcare EDA workbench

A Streamlit-based exploratory data analysis workbench for healthcare activity
data. Users select an analysis workflow, upload one or more historical CSV
files, validate the batch, and open the relevant analysis views.

The project began as a PAH-specific elective-care dashboard. It is now being
generalised so that compatible hospital and provider data can be analysed
through the same workflow.

## Current workflows

### NHS England CSV workflows

These workflows are intended for historical NHS England CSV extracts:

- RTT — waiting-list, backlog, flow, and pathway analysis
- Referrals — demand, source, priority, geography, and specialty analysis
- Outpatient activity — activity, clinic, contact type, and pathway analysis
- Inpatient activity — admissions, specialty mix, heatmaps, and length of stay

### Organisation-provided workflows

These areas require user-provided local data and may contain sensitive
operational or financial information:

- Theatre
- Workforce
- Finance

They are represented in the workflow catalogue but are not yet part of the
active upload-to-analysis path.

## User workflow

1. Choose a workflow from the landing page.
2. Upload one or more CSV files.
3. Review file validation, reporting periods, organisations, and warnings.
4. Open the workflow-specific analysis.
5. Filter, compare, inspect data quality, and export findings where supported.

Uploaded data is held in the active Streamlit session. It is not saved into
the repository. No raw NHS datasets are bundled with this project.

## Active navigation

- Upload and validate
- RTT backlog
- Flow
- RTT specialty
- Referrals
- Outpatient activity
- Data quality
- Inpatient activity

The sidebar is workflow-aware. For example, after uploading RTT data it shows
RTT backlog, Flow, RTT specialty, and Data quality. Unrelated workflows are
not shown.

## Run locally

This project is currently run with the Anaconda Python environment:

~~~bash
/opt/anaconda3/bin/streamlit run app.py --server.port 8501
~~~

Then open [http://localhost:8501](http://localhost:8501).

The main dependencies are Streamlit, pandas, Plotly, requests, and the
standard pandas data-processing libraries available in the project environment.

## Run with Docker

Build and start the application with Docker Compose:

~~~bash
docker compose up --build
~~~

Then open [http://localhost:8501](http://localhost:8501). The container
exposes Streamlit's health endpoint and runs as a non-root user. Uploaded files
are held in the active Streamlit session and are not copied into the image.

To stop the container:

~~~bash
docker compose down
~~~

## Project structure

~~~text
app.py                  Navigation shell and workflow-aware sidebar
views/                  Active Streamlit analysis screens
src/ingestion/          Batch CSV ingestion and validation
src/workflows/          Workflow metadata and source guidance
src/data/               Dataset loaders and cleaning functions
src/transforms/         Reusable filtering and transformation logic
src/metrics/            Reusable analytical metrics
docs/                   Product brief and sprint plan
archive/                Legacy PAH-specific pages retained for reference
~~~

## Design direction

The interface follows the Moorhouse UI Standard:

- Data-first, credible layouts
- Moorhouse purple and teal foundations
- Poppins-style hierarchy and consistent spacing
- Clear workflow context and scope
- Accessible controls and status messages
- Colour used as meaning, never as the only signal

## Data safety

- Do not commit API keys, environment files, raw extracts, or private source data.
- Raw data and Excel workbooks are ignored by Git.
- Use synthetic or anonymised fixtures for development and testing.
- Keep organisation, period, source file, and validation status visible when
  working with an uploaded batch.

## Product planning

See:

- [Product brief](docs/PRODUCT.md)
- [Sprint plan](docs/SPRINT_PLAN.md)
- [Agent guidance](AGENTS.md)

