import streamlit as st

from src.workflows.config import WORKFLOWS

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.5rem;
    }
    .mh-eyebrow {
        color: #00ab8e;
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        line-height: 1.2;
    }
    .mh-lead {
        font-size: 1.1rem;
        line-height: 1.55;
        max-width: 72ch;
        white-space: normal;
        overflow-wrap: anywhere;
    }
    .mh-card {
        background: #ffffff;
        border: 1px solid rgba(60,16,83,.12);
        border-radius: 8px;
        padding: 24px;
        height: 270px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .mh-card h3 {
        margin: 4px 0 2px;
        font-size: 1.2rem;
        line-height: 1.25;
        min-height: 3.1em;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        text-wrap: balance;
    }
    .mh-card p {
        color: #71717a;
        font-size: 1rem;
        line-height: 1.45;
        margin: 0;
        overflow-wrap: anywhere;
    }
    .mh-description {
        min-height: 4.4em;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    .mh-data-source {
        margin-top: auto !important;
        padding-top: 12px;
    }
    @media (max-width: 900px) {
        .mh-card {
            height: auto;
            min-height: 220px;
        }
        .mh-card h3 {
            min-height: unset;
            -webkit-line-clamp: unset;
            display: block;
        }
        .mh-description {
            min-height: unset;
            -webkit-line-clamp: unset;
            display: block;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mh-eyebrow">MOORHOUSE - DATA WORKBENCH</div>',
    unsafe_allow_html=True,
)
st.title("Explore your healthcare data")
st.markdown(
    '<div class="mh-lead">Choose an analysis workflow, upload historical CSV data, '
    "and move from data quality to clear, evidence-led analysis.</div>",
    unsafe_allow_html=True,
)
if st.button("Upload CSV data", type="primary"):
    st.switch_page("views/0_Upload_Data.py")

st.markdown("### Choose an analysis workflow")
cards_per_row = 3

for row_start in range(0, len(WORKFLOWS), cards_per_row):
    row_workflows = WORKFLOWS[row_start : row_start + cards_per_row]
    columns = st.columns(cards_per_row)

    for column_index, column in enumerate(columns):
        with column:
            if column_index >= len(row_workflows):
                st.empty()
                continue

            workflow = row_workflows[column_index]
            st.markdown(
                f"""
                <div class="mh-card">
                    <div class="mh-eyebrow">{workflow.label}</div>
                    <h3>{workflow.label}</h3>
                    <p class="mh-description">{workflow.description}</p>
                    <p class="mh-data-source"><strong>{workflow.data_source}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Start {workflow.label} analysis",
                key=f"start_{workflow.key}",
                use_container_width=True,
            ):
                st.session_state["selected_workflow"] = workflow.key
                st.switch_page("views/0_Upload_Data.py")
