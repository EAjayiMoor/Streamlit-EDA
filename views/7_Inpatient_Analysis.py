import pandas as pd
import plotly.express as px
import streamlit as st

from src.data.inpatient_loader import load_inpatient_data, validate_inpatient_data
from src.transforms.inpatient_transform import (
    filter_inpatients,
    summarise_inpatients_by_month,
)


st.title("Inpatient activity")
st.caption(
    "Admissions, specialty mix, pathway characteristics, and length-of-stay analysis."
)

uploaded_frames = st.session_state.get("eda_frames", [])
uploaded_workflow = st.session_state.get("eda_workflow")

try:
    if uploaded_workflow == "inpatient" and uploaded_frames:
        inpatient_df = pd.concat(uploaded_frames, ignore_index=True)
        st.info("Using the validated inpatient batch from the upload workspace.")
    else:
        inpatient_df = load_inpatient_data()
except Exception as exc:
    st.error(f"Could not load inpatient data: {exc}")
    st.stop()

for warning in validate_inpatient_data(inpatient_df):
    st.warning(warning)

with st.sidebar:
    st.header("Inpatient filters")
    min_date = inpatient_df["Admission datetime"].min().date()
    max_date = inpatient_df["Admission datetime"].max().date()
    selected_dates = st.date_input(
        "Admission date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates
    else:
        start_date, end_date = min_date, max_date

    specialties = sorted(inpatient_df["Standardised_Specialty"].dropna().unique())
    selected_specialties = st.multiselect("Specialty", specialties)

    elective_options = sorted(
        inpatient_df["Elective/emergency"].dropna().unique()
    ) if "Elective/emergency" in inpatient_df.columns else []
    selected_elective = st.multiselect("Elective or emergency", elective_options)

    status_options = sorted(
        inpatient_df["Status"].dropna().unique()
    ) if "Status" in inpatient_df.columns else []
    selected_status = st.multiselect("Status", status_options)

filtered_df = filter_inpatients(
    inpatient_df,
    start_date=start_date,
    end_date=end_date,
    specialties=selected_specialties,
    elective_emergency=selected_elective,
    statuses=selected_status,
)

monthly_df = summarise_inpatients_by_month(filtered_df)
specialty_df = (
    filtered_df.groupby("Standardised_Specialty")["Spell ID"]
    .nunique()
    .reset_index(name="Admissions")
    .sort_values("Admissions", ascending=False)
)

kpis = st.columns(4)
kpis[0].metric("Admissions", f"{filtered_df['Spell ID'].nunique():,.0f}")
kpis[1].metric("Specialties", f"{filtered_df['Standardised_Specialty'].nunique():,.0f}")
kpis[2].metric("Months", f"{filtered_df['Admission_Month'].nunique():,.0f}")
if "LoS" in filtered_df.columns:
    kpis[3].metric("Median length of stay", f"{filtered_df['LoS'].median():,.1f} days")
else:
    kpis[3].metric("Median length of stay", "Not available")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Activity overview", "Specialty mix", "Pathway heatmap", "Length of stay"]
)

with tab1:
    st.subheader("Monthly admissions")
    st.plotly_chart(
        px.line(
            monthly_df,
            x="Admission_Month",
            y="Inpatient Activity",
            markers=True,
            title="Admissions over time",
        ),
        use_container_width=True,
    )
    st.dataframe(monthly_df, use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Admissions by specialty")
    st.plotly_chart(
        px.bar(
            specialty_df.head(30),
            x="Admissions",
            y="Standardised_Specialty",
            orientation="h",
            title="Top specialties by admissions",
        ),
        use_container_width=True,
    )
    st.dataframe(specialty_df, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Admissions by month and specialty")
    heatmap_df = (
        filtered_df.groupby(["Admission_Month", "Standardised_Specialty"])["Spell ID"]
        .nunique()
        .reset_index(name="Admissions")
        .pivot(
            index="Standardised_Specialty",
            columns="Admission_Month",
            values="Admissions",
        )
        .fillna(0)
    )
    st.plotly_chart(
        px.imshow(
            heatmap_df,
            aspect="auto",
            color_continuous_scale=["#f5f3f6", "#3c1053"],
            labels={"color": "Admissions"},
            title="Specialty admission heatmap",
        ),
        use_container_width=True,
    )

with tab4:
    st.subheader("Length-of-stay profile")
    if "LoS" not in filtered_df.columns:
        st.info("Length-of-stay data is not available in this file.")
    else:
        los_df = filtered_df.dropna(subset=["LoS"])
        st.plotly_chart(
            px.histogram(
                los_df,
                x="LoS",
                nbins=40,
                title="Length of stay distribution",
                labels={"LoS": "Length of stay (days)"},
            ),
            use_container_width=True,
        )
        st.dataframe(
            los_df[["Spell ID", "Standardised_Specialty", "Admission datetime", "LoS"]]
            .sort_values("LoS", ascending=False)
            .head(100),
            use_container_width=True,
            hide_index=True,
        )

if st.button("Upload another batch"):
    st.switch_page("views/0_Upload_Data.py")
