import streamlit as st

from src.ui.theme import inject_moorhouse_theme


st.set_page_config(
    page_title="Moorhouse EDA Workbench",
    page_icon="◆",
    layout="wide",
)
inject_moorhouse_theme()


def build_navigation() -> None:
    selected_workflow = st.session_state.get("selected_workflow")
    has_uploaded_batch = bool(st.session_state.get("eda_frames"))

    navigation_pages = [
        st.Page(
            "views/00_Choose_Workflow.py",
            title="Choose workflow",
            icon=":material/home:",
        ),
        st.Page(
            "views/0_Upload_Data.py",
            title="Upload and validate",
            icon=":material/upload_file:",
        ),
    ]

    if selected_workflow and has_uploaded_batch:
        workflow_pages = {
            "rtt": [
                st.Page("views/1_RTT_Backlog.py", title="RTT backlog"),
                st.Page("views/2_Flow.py", title="Flow"),
                st.Page("views/3_Specialty.py", title="RTT specialty"),
                st.Page("views/6_Data_Quality.py", title="Data quality"),
            ],
            "referrals": [
                st.Page("views/4_Referrals.py", title="Referrals"),
                st.Page("views/6_Data_Quality.py", title="Data quality"),
            ],
            "outpatient": [
                st.Page("views/5_Outpatient.py", title="Outpatient activity"),
                st.Page("views/6_Data_Quality.py", title="Data quality"),
            ],
            "inpatient": [
                st.Page("views/7_Inpatient_Analysis.py", title="Inpatient activity"),
                st.Page("views/6_Data_Quality.py", title="Data quality"),
            ],
        }
        selected_pages = workflow_pages.get(selected_workflow, [])
        if selected_pages:
            navigation_pages.extend(selected_pages)

    st.navigation(navigation_pages, position="sidebar").run()


build_navigation()
