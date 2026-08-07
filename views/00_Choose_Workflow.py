import streamlit as st

from src.workflows.config import WORKFLOWS

st.markdown(
    """
    <style>
    .mh-eyebrow {
        color: #00ab8e; font-size: 1rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .mh-lead {
        font-size: 1.1rem; line-height: 1.55; max-width: none;
        white-space: nowrap;
    }
    .mh-card {
        background: #ffffff; border: 1px solid rgba(60,16,83,.12);
        border-radius: 8px; padding: 24px; height: 190px;
        box-sizing: border-box; display: flex; flex-direction: column;
    }
    .mh-card h3 { margin: 8px 0 6px; font-size: 1.2rem; }
    .mh-card p { color: #71717a; font-size: 1rem; line-height: 1.45; }
    .mh-card p:last-child { margin-top: auto; }
    @media (max-width: 900px) {
        .mh-lead { white-space: normal; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="mh-eyebrow">MOORHOUSE · DATA WORKBENCH</div>',
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
columns = st.columns(3)
for index, workflow in enumerate(WORKFLOWS):
    with columns[index % 3]:
        st.markdown(
            f"""
            <div class="mh-card">
                <div class="mh-eyebrow">{workflow.label}</div>
                <h3>{workflow.label} analysis</h3>
                <p>{workflow.description}</p>
                <p><strong>{workflow.data_source}</strong></p>
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
