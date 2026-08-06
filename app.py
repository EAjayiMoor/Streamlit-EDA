import streamlit as st

st.set_page_config(
    page_title="PAH Elective Analytics",
    layout="wide"
)

st.title("PAH Elective Analytics")

st.write(
    """
    Welcome to the PAH Elective Performance App.

    Use the sidebar to navigate:
    - RTT Backlog → backlog position and trends
    - Flow → demand vs activity
    - Constraints → diagnostics and capacity
    - Specialty → drill-down analysis
    """
)