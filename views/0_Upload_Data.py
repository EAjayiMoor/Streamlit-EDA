import streamlit as st

from src.ingestion.batch import manifest_frame, validate_batch
from src.workflows.config import WORKFLOWS, get_workflow


st.title("Upload CSV data")
st.caption("Add one or more historical files for the selected analysis workflow.")

st.markdown(
    """
    <style>
    .mh-callout {
        background: #f5f3f6; border-left: 4px solid #00ab8e;
        border-radius: 6px; padding: 12px 16px; color: #181018;
        margin: 12px 0 8px;
    }
    .mh-callout strong { color: #3c1053; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.expander("Which data can I upload?", expanded=True):
    st.markdown(
        "RTT, referrals, outpatient activity, and inpatient activity are the "
        "initial workflows intended for historical NHS England CSV extracts. "
        "Theatre, workforce, and finance require organisation-provided files "
        "and may contain sensitive operational or financial information."
    )
    st.table(
        [
            {
                "Workflow": workflow.label,
                "Expected source": workflow.data_source,
                "Note": workflow.source_note,
            }
            for workflow in WORKFLOWS
        ]
    )

selected_workflow = st.session_state.get("selected_workflow")
workflow_keys = [workflow.key for workflow in WORKFLOWS]

if selected_workflow in workflow_keys:
    workflow_key = selected_workflow
    st.markdown(f"**Selected workflow:** {get_workflow(workflow_key).label}")
    if st.button("Change workflow"):
        st.session_state.pop("selected_workflow", None)
        st.switch_page("views/00_Choose_Workflow.py")
else:
    workflow_key = st.selectbox(
        "Choose an analysis workflow",
        options=workflow_keys,
        index=workflow_keys.index("rtt"),
        format_func=lambda key: get_workflow(key).label,
    )
workflow = get_workflow(workflow_key)
st.session_state["selected_workflow"] = workflow_key
st.markdown(
    f'<div class="mh-callout"><strong>{workflow.label}</strong> — '
    f'{workflow.description} Source: {workflow.data_source}.</div>',
    unsafe_allow_html=True,
)
st.caption(workflow.source_note)

uploaded_files = st.file_uploader(
    "Upload CSV files",
    type=["csv"],
    accept_multiple_files=True,
    help="You can upload multiple reporting periods or organisations in one batch.",
)

if uploaded_files:
    frames, results = validate_batch(uploaded_files, workflow)
    st.session_state["eda_workflow"] = workflow.key
    st.session_state["eda_frames"] = frames
    st.session_state["eda_manifest"] = manifest_frame(results)

    st.subheader("Validation summary")
    ready = sum(result.status == "Ready" for result in results)
    warnings = sum(result.status == "Warning" for result in results)
    errors = sum(result.status == "Error" for result in results)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Files ready", ready)
    metric_columns[1].metric("Warnings", warnings)
    metric_columns[2].metric("Errors", errors)

    st.dataframe(
        st.session_state["eda_manifest"],
        use_container_width=True,
        hide_index=True,
    )

    if errors:
        st.error("Resolve file errors before continuing.")
    elif warnings:
        st.warning("Review the warnings before using these files for analysis.")
    else:
        st.success("The batch is ready for analysis.")

    if frames:
        with st.expander("Preview the first uploaded file"):
            st.dataframe(frames[0].head(20), use_container_width=True)
        analysis_pages = {
            "rtt": ("views/1_RTT_Backlog.py", "Open RTT backlog analysis"),
            "referrals": ("views/4_Referrals.py", "Open referral analysis"),
            "outpatient": ("views/5_Outpatient.py", "Open outpatient analysis"),
            "inpatient": ("views/7_Inpatient_Analysis.py", "Open inpatient analysis"),
        }
        if workflow.key in analysis_pages and not errors:
            page_path, page_label = analysis_pages[workflow.key]
            if st.button(page_label, type="primary"):
                st.switch_page(page_path)
else:
    st.markdown(
        '<div class="mh-callout">Upload one or more CSV files to begin validation.</div>',
        unsafe_allow_html=True,
    )

st.divider()
if st.button("Back to workflow selection"):
    st.session_state.pop("selected_workflow", None)
    st.session_state.pop("eda_frames", None)
    st.session_state.pop("eda_manifest", None)
    st.switch_page("views/00_Choose_Workflow.py")
