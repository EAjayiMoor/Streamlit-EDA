import streamlit as st

from src.workflows.config import WORKFLOWS


st.markdown(
    """
    <style>
    :root { --mh-brand: #3c1053; --mh-accent: #00ab8e; --mh-bg: #fbfafb; }
    .stApp { background: var(--mh-bg); }
    h1, h2, h3 { color: var(--mh-brand); letter-spacing: -0.01em; }
    .mh-eyebrow {
        color: var(--mh-accent); font-size: 0.8rem; font-weight: 600;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .mh-lead { font-size: 1.1rem; line-height: 1.55; max-width: 72ch; }
    .mh-card {
        background: white; border: 1px solid rgba(60,16,83,.12);
        border-radius: 8px; padding: 24px; height: 190px;
        box-sizing: border-box; display: flex; flex-direction: column;
    }
    .mh-card h3 { margin: 8px 0 6px; font-size: 1.2rem; }
    .mh-card p { color: #71717a; font-size: 1rem; line-height: 1.45; }
    .mh-card p:last-child { margin-top: auto; }
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
